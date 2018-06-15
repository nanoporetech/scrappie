// Homopolymer path calculations
// June 2018
// homopolymer_path() in this file is called by scrappie_raw.c
// from the function _raw_basecall_info calculate_post().
// command line option '--homopolymer' in scrappie_raw.c  should be set to
// 'mean' in order to invoke the mean homopolymer length calculation
// setting the option to 'nochange' (the default) does nothing.

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "homopolymer.h"
// Note that in Scrappie, and for 5-mer blocks,
// index in posteriors is shifted along one step from index
// in path, so that post[t*post->data.f+j] corresponds to path[t+1]
// (Homopolymer path calculations are implemented only for 5-mer blocks)
#define POSTSHIFT(posterior,t,i) ( posterior->data.f[ ((t-1)*(posterior->stride)) + (i) ]  )

const int STAYPATH = -1;        //integer used to represent stay in path

/**  Calculate the 4^p digit in the base-4 representation of the integer x
 *
 *   (for example, base4(3,0)=3, base4(64,3)=1 )
 *
 *   @param x          integer 
 *   @param position   position 0 is the right-hand (least sig) digit
 *
 *   @return an integer in the range 0,1,2,3 representing a base (ACTG)
 **/
int base4(int x, int position) {
    for (int i = 0; i < position; i++)
        x = x / 4;           //passed by value so OK to do this
    return x % 4;
}

/**  Calculate an int that represents b repeated nrep times in base 4
 *
 *   (for example, repeatblock(1,1) = 1
 *                 repeatblock(2,3) = 2 + 4x2 + 16x2  = 44
 *
 *   @param b          integer in range 0 to 3 inclusive representing base
 *   @param reps       number of repeats
 *
 *   @return           int representing b repeated nrep times in base 4
 **/
int repeatblock(int b,int nrep){
    int y=0;
    for(int n = 0; n < nrep; n++)
        y = y * 4 + b;
    return y;
}

/**  Four to the power n
 * 
 *   @param n          integer
 *
 *   @return           4^n
 **/
int fourTo(int n){
    return 1 << (2*n);} 


/**  Find candidate homopolymer runs
 *   findRunsByBase returns vectors containing the
 *   starts and lengths  and bases of homopolymer runs,
 *   defined as segments that
 *   1. (a) start with    [ XYYYY followed by YYYYY or stay , where X!=Y]
 *      (b) or start with [ ZXYYY followed by YYYYY, where X!=Y ] (this is a skip to a homoblock:Z=X is allowed)
 *      (c) or start with [ ZXYYY followed by any number of stays and then YYYYY, where X!=Y ]
 *      (the last possibility is a skip to a homoblock at the first YYYYY - so the ambiguous part
 *      starts at the YYYYY)
 *   2. end with something that is not (YYYYY or stay)
 *
 *   The start is the location of the first (YYYYY or stay) in case 1a, and
 *   the location of the first YYYYY in case 1b or 1c.
 *   The length is the number of successive occurrences of (YYYYY or stay),
 *   starting at the point identified.
 *   The 'base' returned is an integer 0 to 3 representing A(0),C(1),G(2), or T(3).
 *   Note that space for contents of runstarts, runlengths is allocated by this function.
 *
 *   @param path          vector of ints representing path (ass
 *   @param runstarts     when function returns, runstarts contains pointer to array which has start location of each homopol run found
 *   @param runlengths    similarly points to array of lengths in the path of each homopol run segment (this is not the same as the run length)
 *   @param runbases      similarly points to array giving the bases that repeat (coded as integers 0123 = ACGT)
 *   @param pathlength    length of the array 'path'
 *
 *   @return number of homopolymer runs found
 **/
int findRuns(int *path, int **runstarts, int **runlengths, int **runbases,
             int pathlength, int kmerlength) {
    //Allocate half path length to each of runstarts, runlengths, runbases.
    //We'll reduce this size before passing back to caller.
    const int vecsize = pathlength / 2;
    const int fkm1 = fourTo(kmerlength-1);  //Numbers that will be needed repeatedly in calculations
    const int fkm2 = fourTo(kmerlength-2);
    *runstarts  = (int *)malloc(vecsize * sizeof(int));
    *runlengths = (int *)malloc(vecsize * sizeof(int));
    *runbases   = (int *)malloc(vecsize * sizeof(int));
    int runcount = 0;
    for (int base = 0; base < 4; base++)        //Bases ACGT=0123
    {
        int repeatk   = repeatblock(base,kmerlength);          //Index of base repeated k times where k is kmer length
        int repeatkm1 = repeatblock(base,kmerlength-1);        //Repeated (k-1) times
        int repeatkm2 = repeatblock(base,kmerlength-2);        //Repeated (k-2) times
        for (int i = 1; i < pathlength - 2; i++)               //Location of start in path
        {
            int p = path[i - 1];        //Just to provide shorthand
            int q = path[i];
            //Search for elements that go (XYYYY followed by YYYYY or stay) - 1a above
            //Don't include X=Y. Exclude -1 at prev because its remainder is the same as TTTT            
            if ((p % fkm1 == repeatkm1) && (p != repeatk) && (p != STAYPATH)
                && ((q == STAYPATH) || (q == repeatk))) {
                //Hunt for the first location that isn't stay or repeatk: this is the end of the run
                int e = i + 1;
                while (e < pathlength
                       && (path[e] == STAYPATH || path[e] == repeatk))
                    e++;
                (*runstarts)[runcount] = i;     //Location of start of run
                (*runlengths)[runcount] = e - i;
                (*runbases)[runcount] = base;
                runcount++;
            }
            //Search for elements that go (ZXYYY followed by zero or more stays then YYYYY) - 1bc above
            //Don't include X=Y. Exclude -1 at prev because its remainder is the same as TTTT            
            if ((p % fkm2 == repeatkm2) && (p % fkm1 != repeatkm1) && (p != STAYPATH)
                && ((q == STAYPATH) || (q == repeatk))) {
                //Hunt for the first location that isn't stay after (not including) (i-1)
                int j = i;
                while (j < pathlength && path[j] == STAYPATH)
                    j++;
                //So far we have (ZXYYY followed by zero or more stays then something)
                //If the something is YYYYY and we still have any space left before the end...
                if (path[j] == repeatk && j < pathlength - 1) {
                    //Hunt for the first location that isn't stay or repeat5: this is the end of the run
                    int e = j + 1;
                    while (e < pathlength
                           && (path[e] == -1 || path[e] == repeatk))
                        e++;
                    (*runstarts)[runcount] = j; //Location of start of run: note this is j not i
                    (*runlengths)[runcount] = e - j;
                    (*runbases)[runcount] = base;
                    runcount++;
                }
            }
        }
    }
    //We allocated the arrays at the start using far too much space: reallocate to what we need
    *runstarts  = (int *)realloc(*runstarts,  runcount * sizeof(int));
    *runlengths = (int *)realloc(*runlengths, runcount * sizeof(int));
    *runbases   = (int *)realloc(*runbases,   runcount * sizeof(int));
    return runcount;
}

/**  Find homopolymer runs in a path array and modify according to path calculations
 *   Details: hunt for homopolymer runs, calculate the mean run length using the posterior matrix,
 *   modify the given path so that instead of the Viterbi run length it has the mean
 *   run length.
 *   The array viterbipath is modified by this function so that after a call, it contains
 *   the corrected path (with mean run lengths rather than Viterbi ones).
 *   If pathCalculationFlag is set to HOMOPOLYMER_NOCHANGE we do no modifications.
 *   If pathCalculationFlag is HOMOPOLYMER_MEAN  we do the mean calculation.
 *   Future changes may allow other modes of operation, determined by this flag.
 *
 *   @param post                     scrappie matrix of posterior probabilities (logged)
 *   @param viterbipath              vector of ints representing path
 *   @param pathCalculationFlag      set to HOMOPOLYMER_NOCHANGE or HOMOPOLYMER_MEAN - see docstring above
 *
 *   @return 
 **/
int homopolymer_path(scrappie_matrix post, int *viterbipath,
                     enum homopolymer_calculation pathCalculationFlag) {
    if (pathCalculationFlag == HOMOPOLYMER_NOCHANGE) {
        return 0;
    }
    const int nblock = post->nc;        //Number of locations in read
    const int staystate = post->nr-1;     //index of the stay in posterior vectors (also the last element)
    const int kmerlength = (int) (logf((float)(post->nr)) / logf(4.0f)); // base 4 log of nblock gives us kmer length
    int *runstarts;             //Location of start (first ambiguous location) in each homopol run
    int *runlengths;            //Length of ambiguous section
    int *runbases;              //Integer 0-3 representing the base which repeats in the homopol run
    //Find homopolymer runs
    int runcount =
        findRuns(viterbipath, &runstarts, &runlengths, &runbases, nblock, kmerlength);

    for (int nrun = 0; nrun < runcount; nrun++) //For each homopolymer run...
    {
        //Calculate Viterbi (as a check against the existing sequence) and mean numbers of non-stays
        //While we're at it, count number of non-stays in the existing sequence
        int nviterbi = 0;
        int ncalcviterbi = 0;
        double nmean = 0.0;
        int runstate = runbases[nrun] * 341;    //index of a fivemer with the repeat base repeated 5x
        int runlength = runlengths[nrun];
        int ambigfrom = runstarts[nrun];        //location of the first ambiguous block in the homopol run
        int ambigto = ambigfrom + runlength - 1;        //location of the last ambiguous block
        //Calculate normalised stay probabilities for each of the ambiguous locations
        for (int i = ambigfrom; i <= ambigto; i++) {
            double psu = expf(POSTSHIFT(post, i, staystate));   //Stay probability (un-normalised) - note posts are logged in Scrappie and shifted one step
            double pru = expf(POSTSHIFT(post, i, runstate));    //Repeat block probability (un-normalised)
            double pr = pru / (pru + psu);      //Normalised repeat block probability
            nmean = nmean + pr;
            if (pr > 0.5)
                ncalcviterbi = ncalcviterbi + 1;
            if (viterbipath[i] == runstate)
                nviterbi = nviterbi + 1;
        }
        int newn = (int)(nmean + 0.5);  //nmean is a float so need to round
        //Make modification to the path if necessary
        if (newn != nviterbi) {
            for (int i = 0; i <= ambigto - ambigfrom; i++) {
                //Fill in the right number of repeat blocks, putting stays for the rest
                //(order doesn't matter since this path will be collapsed to a sequence)
                if (i < newn) {
                    viterbipath[i + ambigfrom] = runstate;
                } else
                    viterbipath[i + ambigfrom] = STAYPATH;
            }
        }
    }
    free(runstarts);
    free(runlengths);
    free(runbases);
    return 0;
}
