class LogArgumentsError(ValueError):
    """Raised when the logs command receives invalid arguments."""


def parse_log_arguments(arguments: list[str]) -> tuple[str | None, int]:
    """Return (service_id, line_count) for the ./ai logs arguments."""
    service_id = None
    line_count = 40
    lines_were_set = False
    index = 0

    while index < len(arguments):
        argument = arguments[index]

        if argument in ("-n", "--lines") or argument.startswith("--lines="):
            if lines_were_set:
                raise LogArgumentsError("Specify --lines only once.")
            lines_were_set = True

            if argument.startswith("--lines="):
                value = argument.partition("=")[2]
                index += 1
            else:
                if index + 1 >= len(arguments):
                    raise LogArgumentsError(
                        "--lines must be followed by a positive integer."
                    )
                value = arguments[index + 1]
                index += 2

            if not value.isascii() or not value.isdecimal():
                raise LogArgumentsError(
                    "--lines must be a positive integer."
                )

            line_count = int(value)
            if line_count <= 0:
                raise LogArgumentsError(
                    "--lines must be a positive integer."
                )
            continue

        if argument.startswith("-"):
            raise LogArgumentsError(f"Unknown logs option: {argument}")

        if service_id is not None:
            raise LogArgumentsError(
                "Only one service ID may be specified."
            )

        service_id = argument
        index += 1

    return service_id, line_count
