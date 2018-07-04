#include <stdio.h>

#include "scrappie_licence.h"
#include "scrappie_stdlib.h"
#include "scrappie_subcommands.h"

char *help_options[2] = { NULL, "--help" };

static const char *scrappie_help_header =
    "Scrappie is a technology demonstrator for the Oxford Nanopore Technologies\n"
    "Limited Research Algorithms group.\n";

static const char *scrappie_help_footer =
    "This project began life as a proof (bet) that a base caller could be written\n"
    "in a low level language in under 8 hours.  Some of the poor and just plain odd\n"
    "design decisions, along with the lack of documentation, are a result of its\n"
    "inception. In keeping with ONT's fish naming policy, the project was originally\n"
    "called Crappie.\n"
    "\n"
    "Scrappie's purpose is to demonstrate the next generation of base calling and, as\n"
    "such, may change drastically between releases and breaks backwards\n"
    "compatibility.  Newer versions may drop support of current features or change their\n"
    "behaviour.\n";

extern const char *argp_program_version;

int main_help_short(void) {
    int ret = fputs(scrappie_help_header, stdout);
    if (EOF != ret) {
        fputc('\n', stdout);
        ret = fprint_scrappie_commands(stdout, true);
    }
    return (EOF != ret) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main_help(int argc, char *argv[]) {

    if (argc == 1) {
        int ret = fputs(scrappie_help_header, stdout);
        if (EOF != ret) {
            fputc('\n', stdout);
            ret = fprint_scrappie_commands(stdout, true);
        }
        if (EOF != ret) {
            fputc('\n', stdout);
            ret = fputs(scrappie_help_footer, stdout);
        }
        return (EOF != ret) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    int ret = EXIT_FAILURE;
    switch (get_scrappie_mode(argv[1])) {
    case SCRAPPIE_MODE_HELP:
        printf("Recursively calling scrappie help\n");
        break;
    case SCRAPPIE_MODE_EVENTS:
        help_options[0] = argv[1];
        ret = main_events(2, help_options);
        break;
    case SCRAPPIE_MODE_RAW:
        help_options[0] = argv[1];
        ret = main_raw(2, help_options);
        break;
    case SCRAPPIE_MODE_SQUIGGLE:
        help_options[0] = argv[1];
        ret = main_squiggle(2, help_options);
        break;
    case SCRAPPIE_MODE_MAPPY:
        help_options[0] = argv[1];
        ret = main_mappy(2, help_options);
        break;
    case SCRAPPIE_MODE_SEQMAPPY:
        help_options[0] = argv[1];
        ret = main_seqmappy(2, help_options);
        break;
    case SCRAPPIE_MODE_EVENT_TABLE:
        help_options[0] = argv[1];
        ret = main_event_table(2, help_options);
        break;
    default:
        ret = EXIT_FAILURE;
        warnx("Unrecognised subcommand %s\n", argv[1]);
    }

    return ret;
}

int main_licence(int argc, char *argv[]) {
    int ret = fputs(scrappie_licence_text, stdout);
    return (EOF != ret) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main_version(int argc, char *argv[]) {
    int ret = puts(argp_program_version);
    return (EOF != ret) ? EXIT_SUCCESS : EXIT_FAILURE;
}
