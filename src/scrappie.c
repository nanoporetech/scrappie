#include <err.h>
#include <stdlib.h>

#include "scrappie_licence.h"
#include "scrappie_subcommands.h"
#include "version.h"

#if !defined(SCRAPPIE_VERSION)
#define SCRAPPIE_VERSION "unknown"
#endif

#include <argp.h>


const char * argp_program_version = "scrappie " SCRAPPIE_VERSION;
const char * argp_program_bug_address = "<tim.massingham@nanoporetech.com>";
static struct argp_option options[] = {
	{"licence", 1, 0, 0, "Print licensing information"},
	{"license", 2, 0, OPTION_ALIAS, "Print licensing information"},
	{0}
};
static char doc[] = "Scrappie basecaller";
static char args_doc[] = "";
struct arguments {
	int argc;
	char ** argv;
};
static struct arguments args = {0, NULL};

static error_t parse_arg(int key, char * arg, struct  argp_state * state){
	switch(key){
		int ret = 0;
	case 1:
	case 2:
		ret = fputs(scrappie_licence_text, stdout);
		exit((EOF != ret) ? EXIT_SUCCESS : EXIT_FAILURE);
		break;
	case ARGP_KEY_ARG:
		args.argv= &state->argv[state->next - 1];
		args.argc = state->argc - state->next + 1;
		state->next = state->argc;
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}

	return 0;
}

static struct argp argp = {options, parse_arg, args_doc, doc};

int main(int argc, char * argv[]){

	argp_parse(&argp, argc, argv, 0, 0, NULL);

	if(args.argc == 0){
		// Called as program name on it's own
		return main_help_short();
	}

	int ret = EXIT_FAILURE;
	switch(get_scrappie_mode(args.argv[0])){
		case SCRAPPIE_MODE_HELP:
			ret = main_help(args.argc, args.argv);
			break;
		case SCRAPPIE_MODE_EVENTS:
			ret = main_events(args.argc, args.argv);
			break;
		case SCRAPPIE_MODE_RAW:
			ret = main_raw(args.argc, args.argv);
			break;
		case SCRAPPIE_MODE_LICENCE:
			ret = main_licence(args.argc, args.argv);
			break;
		case SCRAPPIE_MODE_VERSION:
			ret = main_version(args.argc, args.argv);
			break;
		default:
			ret = EXIT_FAILURE;
			warnx("Unrecognised subcommand %s\n", argv[1]);
	}

	return ret;
}

