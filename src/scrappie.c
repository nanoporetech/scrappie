#include <err.h>
#include "scrappie_stdlib.h"

#include "scrappie_subcommands.h"
#include "version.h"

#if !defined(SCRAPPIE_VERSION)
#    define SCRAPPIE_VERSION "unknown"
#endif
const char *argp_program_version = "scrappie " SCRAPPIE_VERSION;
const char *argp_program_bug_address = "<tim.massingham@nanoporetech.com>";

int main(int argc, char *argv[]) {

    if (argc == 1) {
        // Called as program name on it's own
        return main_help_short();
    }

    int ret = EXIT_FAILURE;
    switch (get_scrappie_mode(argv[1])) {
    case SCRAPPIE_MODE_HELP:
        ret = main_help(argc - 1, argv + 1);
        break;
    case SCRAPPIE_MODE_EVENTS:
        ret = main_events(argc - 1, argv + 1);
        break;
    case SCRAPPIE_MODE_RAW:
        ret = main_raw(argc - 1, argv + 1);
        break;
    case SCRAPPIE_MODE_LICENCE:
        ret = main_licence(argc - 1, argv + 1);
        break;
    case SCRAPPIE_MODE_VERSION:
        ret = main_version(argc - 1, argv + 1);
        break;
    case SCRAPPIE_MODE_SQUIGGLE:
        ret = main_squiggle(argc - 1, argv + 1);
        break;
    case SCRAPPIE_MODE_MAPPY:
        ret = main_mappy(argc - 1, argv + 1);
        break;
    case SCRAPPIE_MODE_SEQMAPPY:
        ret = main_seqmappy(argc - 1, argv + 1);
        break;
    case SCRAPPIE_MODE_EVENT_TABLE:
        ret = main_event_table(argc - 1, argv + 1);
        break;
    default:
        ret = EXIT_FAILURE;
        warnx("Unrecognised subcommand %s\n", argv[1]);
    }

    return ret;
}
