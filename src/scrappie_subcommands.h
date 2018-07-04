#pragma once
#ifndef SCRAPPIE_SUBCOMMANDS_H
#    define SCRAPPIE_SUBCOMMANDS_H

#    include <stdbool.h>
#    include <stdio.h>

// Helper functions for subcommmads
//
enum scrappie_mode {SCRAPPIE_MODE_EVENTS = 0,
                    SCRAPPIE_MODE_HELP,
                    SCRAPPIE_MODE_LICENCE,
                    SCRAPPIE_MODE_RAW,
                    SCRAPPIE_MODE_VERSION,
                    SCRAPPIE_MODE_SQUIGGLE,
                    SCRAPPIE_MODE_MAPPY,
                    SCRAPPIE_MODE_SEQMAPPY,
                    SCRAPPIE_MODE_EVENT_TABLE,
                    SCRAPPIE_MODE_INVALID };
static const enum scrappie_mode scrappie_ncommand = SCRAPPIE_MODE_INVALID;

enum scrappie_mode get_scrappie_mode(const char *modestr);
const char *scrappie_mode_string(const enum scrappie_mode mode);
const char *scrappie_mode_description(const enum scrappie_mode mode);
int fprint_scrappie_commands(FILE * fp, bool header);

// Main routines for subcommands
int main_events(int argc, char *argv[]);
int main_event_table(int argc, char *argv[]);
int main_help(int argc, char *argv[]);
int main_help_short(void);
int main_licence(int argc, char *argv[]);
int main_mappy(int argc, char * argv[]);
int main_raw(int argc, char *argv[]);
int main_seqmappy(int argc, char * argv[]);
int main_squiggle(int argc, char * argv[]);
int main_version(int argc, char *argv[]);

#endif                          /* SCRAPPIE_SUBCOMMANDS_H */
