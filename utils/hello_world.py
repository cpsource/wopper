import os
from dotenv import load_dotenv
from logger import get_logger

log = get_logger(__name__)
log.debug("Starting hello_world.py")

def print_hello_world():
    """Function to print 'hello world' to the console."""
    log.info("hello world")

def main():
    """Main function to test the print_hello_world function."""
    print_hello_world()

if __name__ == "__main__":
    main()
