
import asyncio
from ghost_agent.tools.system import tool_check_health

async def test_healthcheck():
    print("Testing tool_check_health...")
    try:
        result = await tool_check_health(context=None)
        print("Healthcheck Result:\n" + result)
        if "System Status: Online" in result:
            print("SUCCESS: Healthcheck returned expected status.")
        else:
            print("FAILURE: Healthcheck result did not contain expected status.")
    except Exception as e:
        print(f"FAILURE: Healthcheck raised an exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_healthcheck())
