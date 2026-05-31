import asyncio
from pydantic import BaseModel, ConfigDict, field_validator

class ImpossibleSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    magic_number: int

    @field_validator('magic_number')
    @classmethod
    def must_be_exactly_42_times_pi(cls, v: int) -> int:
        raise ValueError('This validator always fails for testing purposes')

async def test_error():
    from src.llm.instructor_client import InstructorClient, StructuredGenerationError
    client = InstructorClient(max_retries=1)  # only 1 retry to keep it fast
    try:
        await client.create_structured('Return magic_number=99', ImpossibleSchema)
        print('FAIL: should have raised StructuredGenerationError')
    except StructuredGenerationError as e:
        print('CORRECT: StructuredGenerationError raised, hash=', e.prompt_hash)
    finally:
        await client.close()

asyncio.run(test_error())
