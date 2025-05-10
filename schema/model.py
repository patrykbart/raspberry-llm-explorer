from pydantic import BaseModel, Field
from pydantic.json_schema import JsonSchemaValue

MODEL_NAME = "llava-custom"
PORT = 12345
RASPBERRY_PI_IP = "192.168.137.147"
RASPBERRY_PI_PORT = 5053

SYSTEM_PROMPT = """You are an autonomous navigation controller for a car. Based on the camera image, your task is to decide the car's next movement to avoid obstacles and explore the world. The image you get always shows your latest location. Your main obiective is to explore and have fun. JSPN commands:
- "m": movement command. Use when not turning ("F" for forward, "B" for backward, "L" for left, "R" for right, "S" for stop),
- "s": speed as a percentage (0-100),
- "t": turn angle in degrees (0 if not turning),
- "d": duration in seconds (0-4).
- "r": sentence describing what you see and why you made this decision

Output exactly one sentence and then valid JSON object."""

class CarMovementCommand(BaseModel):
    m: str = Field(..., description="Movement command", enum=["F", "B", "L", "R", "S"])
    s: float = Field(..., ge=0, le=100)
    t: float = Field(..., ge=0, le=360)
    d: float = Field(..., ge=0, le=4)
    r: str = Field(...)

command_schema: JsonSchemaValue = CarMovementCommand.model_json_schema()