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

    int ret = main_squiggle(argc, argv);

    return ret;
}
