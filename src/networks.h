#ifndef NETWORKS_H
#define NETWORKS_H

#include <stdbool.h>
#include "events.h"

Mat_rptr nanonet_posterior(const event_table events, int trim, float min_prob, bool return_log);

#endif  /* NETWORKS_H */


