"""XSD-based validator for JUnit XML produced by the untrusted runner.

Treats *every* XML input as untrusted: rejects DOCTYPE, external entities,
non-standard root elements, and unexpected children. Uses ``lxml`` for XSD
validation and ``defusedxml.lxml`` for the initial parse so XXE / billion-laughs
attacks cannot succeed.

CLI:
    python -m ci.validators.xml_schema_validator <path_to_xml>

Exit codes:
    0  — XML is structurally safe and conforms to the JUnit + properties schema
    1  — XML rejected (full reason on stderr; raw XML body never logged)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import defusedxml.ElementTree as defused_etree
from defusedxml.common import DefusedXmlException
from lxml import etree

logger = logging.getLogger(__name__)

# XSD covering JUnit XML produced by pytest --junitxml, plus the
# <properties>/<property> sub-tree used to embed AI telemetry. Anything outside
# this schema is treated as a poisoned payload.
_JUNIT_XSD: Final[str] = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" elementFormDefault="unqualified">

  <xs:simpleType name="nonNegativeIntStr">
    <xs:restriction base="xs:string"><xs:pattern value="[0-9]+"/></xs:restriction>
  </xs:simpleType>

  <xs:complexType name="propertyType">
    <xs:attribute name="name" type="xs:string" use="required"/>
    <xs:attribute name="value" type="xs:string" use="required"/>
  </xs:complexType>

  <xs:complexType name="propertiesType">
    <xs:sequence>
      <xs:element name="property" type="propertyType" minOccurs="0" maxOccurs="unbounded"/>
    </xs:sequence>
  </xs:complexType>

  <xs:complexType name="failureType" mixed="true">
    <xs:attribute name="message" type="xs:string" use="optional"/>
    <xs:attribute name="type"    type="xs:string" use="optional"/>
  </xs:complexType>

  <xs:complexType name="skippedType" mixed="true">
    <xs:attribute name="message" type="xs:string" use="optional"/>
    <xs:attribute name="type"    type="xs:string" use="optional"/>
  </xs:complexType>

  <xs:complexType name="testcaseType">
    <xs:sequence>
      <xs:element name="properties" type="propertiesType" minOccurs="0" maxOccurs="1"/>
      <xs:element name="skipped"    type="skippedType"    minOccurs="0" maxOccurs="1"/>
      <xs:element name="failure"    type="failureType"    minOccurs="0" maxOccurs="unbounded"/>
      <xs:element name="error"      type="failureType"    minOccurs="0" maxOccurs="unbounded"/>
      <xs:element name="system-out" type="xs:string"      minOccurs="0" maxOccurs="1"/>
      <xs:element name="system-err" type="xs:string"      minOccurs="0" maxOccurs="1"/>
    </xs:sequence>
    <xs:attribute name="name"      type="xs:string"          use="required"/>
    <xs:attribute name="classname" type="xs:string"          use="optional"/>
    <xs:attribute name="time"      type="xs:string"          use="optional"/>
    <xs:attribute name="file"      type="xs:string"          use="optional"/>
    <xs:attribute name="line"      type="nonNegativeIntStr"  use="optional"/>
  </xs:complexType>

  <xs:complexType name="testsuiteType">
    <xs:sequence>
      <xs:element name="properties" type="propertiesType" minOccurs="0" maxOccurs="1"/>
      <xs:element name="testcase"   type="testcaseType"   minOccurs="0" maxOccurs="unbounded"/>
      <xs:element name="system-out" type="xs:string"      minOccurs="0" maxOccurs="1"/>
      <xs:element name="system-err" type="xs:string"      minOccurs="0" maxOccurs="1"/>
    </xs:sequence>
    <xs:attribute name="name"      type="xs:string"         use="required"/>
    <xs:attribute name="tests"     type="nonNegativeIntStr" use="optional"/>
    <xs:attribute name="failures"  type="nonNegativeIntStr" use="optional"/>
    <xs:attribute name="errors"    type="nonNegativeIntStr" use="optional"/>
    <xs:attribute name="skipped"   type="nonNegativeIntStr" use="optional"/>
    <xs:attribute name="time"      type="xs:string"         use="optional"/>
    <xs:attribute name="timestamp" type="xs:string"         use="optional"/>
    <xs:attribute name="hostname"  type="xs:string"         use="optional"/>
  </xs:complexType>

  <xs:complexType name="testsuitesType">
    <xs:sequence>
      <xs:element name="testsuite" type="testsuiteType" minOccurs="0" maxOccurs="unbounded"/>
    </xs:sequence>
    <xs:attribute name="name"     type="xs:string"         use="optional"/>
    <xs:attribute name="tests"    type="nonNegativeIntStr" use="optional"/>
    <xs:attribute name="failures" type="nonNegativeIntStr" use="optional"/>
    <xs:attribute name="errors"   type="nonNegativeIntStr" use="optional"/>
    <xs:attribute name="time"     type="xs:string"         use="optional"/>
  </xs:complexType>

  <xs:element name="testsuites" type="testsuitesType"/>
  <xs:element name="testsuite"  type="testsuiteType"/>

</xs:schema>
"""

_ALLOWED_ROOTS: Final[frozenset[str]] = frozenset({"testsuites", "testsuite"})


def _compile_schema() -> etree.XMLSchema:
    schema_doc = etree.fromstring(_JUNIT_XSD.encode("utf-8"))
    return etree.XMLSchema(schema_doc)


_SCHEMA: Final[etree.XMLSchema] = _compile_schema()


def validate_file(xml_path: Path) -> bool:
    """Validate ``xml_path`` against the JUnit + properties XSD.

    Args:
        xml_path: Path to the candidate XML file.

    Returns:
        ``True`` only when the XML is free of DOCTYPE / external entities,
        rooted at ``<testsuites>`` (or ``<testsuite>``), and structurally
        conformant to the embedded XSD. ``False`` otherwise.
    """
    path = Path(xml_path)
    if not path.is_file():
        logger.error("xml_validator: path is not a file path=%s", path)
        return False

    # Step 1 — defusedxml gate: rejects DOCTYPE, external entities, network refs.
    try:
        defused_etree.parse(
            str(path),
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )
    except DefusedXmlException as exc:
        logger.error(
            "xml_validator: defused parse rejected path=%s reason=%s",
            path,
            type(exc).__name__,
        )
        return False
    except Exception as exc:
        logger.error(
            "xml_validator: parse error path=%s reason=%s",
            path,
            type(exc).__name__,
        )
        return False

    # Step 2 — re-parse with a hardened lxml parser for XSD validation.
    safe_parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        dtd_validation=False,
        huge_tree=False,
    )
    try:
        tree = etree.parse(str(path), parser=safe_parser)
    except etree.XMLSyntaxError as exc:
        logger.error(
            "xml_validator: lxml syntax error path=%s reason=%s",
            path,
            type(exc).__name__,
        )
        return False

    root = tree.getroot()
    if etree.QName(root).localname not in _ALLOWED_ROOTS:
        logger.error(
            "xml_validator: non-standard root path=%s root=%s",
            path,
            etree.QName(root).localname,
        )
        return False

    if not _SCHEMA.validate(tree):
        for err in _SCHEMA.error_log:
            logger.error(
                "xml_validator: schema violation path=%s line=%s domain=%s message=%s",
                path,
                err.line,
                err.domain_name,
                err.message,
            )
        return False

    return True


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xml_schema_validator",
        description="Validate JUnit XML safely (treats input as untrusted).",
    )
    parser.add_argument("xml_path", type=Path, help="Path to JUnit XML file to validate.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    ok = validate_file(args.xml_path)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover — exercised via CLI tests
    sys.exit(_main())


__all__ = ["validate_file"]
