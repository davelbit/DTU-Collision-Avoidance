#!/usr/bin/env python3
######################################################################
# Authors:      - Varun_debug Ghatrazu <s210245>
#               - David Parham <s202385>
#
# Course:       Deep learning approaches for damage limitation in car-human collisions
# Semester:     Fall 2021
# Institution:  Technical University of Denmark (DTU)
######################################################################

from image import main

import click
from loguru import logger


@click.command()
@click.option("-d", "--debug", "debug", is_flag=True, help="Use this flag to enable debugging")
def logging(debug: bool) -> None:
    """This function is responsible for enabling and configuring all debugging settings"""

    if debug:
        logger.info("Initialize debug mode...")
        run_debug()
        logger.debug("Initialize logger. Files can be found in the logfiles/ directory")
        logger.add(
            "logfiles/file_{time}.log",
            colorize=True,
            format="<green>{time}</green> <level>{message}</level>",
            backtrace=True,
            diagnose=True,
        )
    else:
        logger.disable("__main__")
        main()


@logger.catch
def run_debug():
    main()


if __name__ == "__main__":
    run_debug()
