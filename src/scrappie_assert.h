#pragma once
#ifndef SCRAPPIE_ASSERT_H
#    define SCRAPPIE_ASSERT_H
#    include <assert.h>
#    include <err.h>
#    include <stdlib.h>

#    ifdef ABORT_ON_NULL
#        define RETURN_NULL_IF(A, B) \
		if (A) {  	\
			fprintf(stderr, "Failure at %s : %d", __FILE__, __LINE__);	\
			abort();	\
		}
#    else
#        define RETURN_NULL_IF(A, B) if (A) { return B; }
#    endif

#endif                          /* SCRAPPIE_ASSERT_H */
