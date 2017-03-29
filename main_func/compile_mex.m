% compile mex's
mex LLT_model.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
mex FISTA_TV.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
mex SplitBregman_TV.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"