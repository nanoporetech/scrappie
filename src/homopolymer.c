//Homopolymer path calculations
// June 2018
// homopolymer_path() in this file is called by scrappie_raw.c
// from the function _raw_basecall_info calculate_post().
// command line option '--homopolymer' in scrappie_raw.c  should be set to
// 'mean' in order to invoke the mean homopolymer length calculation
// setting the option to 'nochange' (the default) does nothing.

#include <stdio.h>
#include <math.h>
#include "homopolymer.h"
//#define DEBUG

//Note that in Scrappie, index in posteriors is shifted along one step from index
//in path, so that post[t*post->data.f+j] corresponds to path[t+1]
#define POSTSHIFT(posterior,t,i) ( posterior->data.f[ ((t-1)*(posterior->stride)) + (i) ]  )
const int STAYPOST=1024; //index of the stay in posterior vectors (also the last element)
const int STAYPATH=-1; //integer used to represent stay in path
const char * BASES="ACGT";


/*Calculate the 4^p digit in the base-4 representation of the integer x
(for example, b4(3,0)=3, b4(64,3)=1 )*/
int base4(int x, int position)
{
    int y=x;
    for(int i=0;i<position;i++)
        y=y/4;
    return y%4;
}

/*Calculate base letter from integer representing fivemer block
and the position in the block (position 0 is the last, position
4 is the first, so position's like the significance of a base-4 digit)*/
char basefromint(int x,int position)
{
    return BASES[base4(x,position)];
}

/*Test function for basefromint()*/
int basefrominttest()
{
    int i=1023;
    for(int pos=0;pos<5;pos++)
        fprintf(stderr,"%d position %d gives %c\n",i,pos,basefromint(i,pos));
    i=2;
    for(int pos=0;pos<5;pos++)
        fprintf(stderr,"%d position %d gives %c\n",i,pos,basefromint(i,pos));
    i=767;
    for(int pos=0;pos<5;pos++)
        fprintf(stderr,"%d position %d gives %c\n",i,pos,basefromint(i,pos));
    return 0;
}


/*Find candidate homopolymer runs
findRunsByBase returns vectors containing the
starts and lengths  and bases of homopolymer runs,
defined as segments that
1. (a) start with    [ XYYYY followed by YYYYY or stay , where X!=Y]
   (b) or start with [ ZXYYY followed by YYYYY, where X!=Y ] (this is a skip to a homoblock:Z=X is allowed)
   (c) or start with [ ZXYYY followed by any number of stays and then YYYYY, where X!=Y ]
   (the last possibility is a skip to a homoblock at the first YYYYY - so the ambiguous part
   starts at the YYYYY)
2. end with something that is not (YYYYY or stay)
The start is the location of the first (YYYYY or stay) in case 1a, and
the location of the first YYYYY in case 1b or 1c.
The length is the number of successive occurrences of (YYYYY or stay),
starting at the point identified.
The 'base' returned is an integer 0 to 3 representing A(0),C(1),G(2), or T(3).*/
int findRuns(int *path,int **runstarts,int **runlengths, int **runbases, int pathlength)
{
    //Allocate half this length to each of runstarts, runlengths, runbases.
    //We'll reduce this size before passing back to caller.
    int vecsize = pathlength/2;
    *runstarts  = (int  *) malloc(vecsize*sizeof(int));
    *runlengths = (int  *) malloc(vecsize*sizeof(int));
    *runbases   = (int  *) malloc(vecsize*sizeof(int));
    int runcount=0;
    for(int base=0;base<4;base++)   //Bases ACGT=0123
    {
        int repeat5 = base * 341;       //Index of base repeated 5x
        int repeat4 = base * 85;        //Repeated 4x
        int repeat3 = base * 21;        //3x
        for(int i=1;i<pathlength-2;i++) //Location of start in path
        {
            int p=path[i-1];//Just to provide shorthand
            int q=path[i];
            //Search for elements that go (XYYYY followed by YYYYY or stay) - 1a above
            //Don't include X=Y. Exclude -1 at prev because its remainder is the same as TTTT            
            if(  (p%256==repeat4)&&(p!=repeat5)&&(p!=STAYPATH)  && ((q==STAYPATH)||(q==repeat5))  )
            {                
                //Hunt for the first location that isn't stay or repeat5: this is the end of the run
                int e=i+1;
                while( e<pathlength && (path[e]==STAYPATH || path[e]==repeat5) ) e++;
                (*runstarts)[runcount]=i;//Location of start of run
                (*runlengths)[runcount]=e-i;
                (*runbases)[runcount]=base;
                runcount++;
            }
            //Search for elements that go (ZXYYY followed by zero or more stays then YYYYY) - 1bc above
            //Don't include X=Y. Exclude -1 at prev because its remainder is the same as TTTT            
            if(  (p%64==repeat3)&&(p%256!=repeat4)&&(p!=STAYPATH)  && ((q==STAYPATH)||(q==repeat5))  )
            {                
                //Hunt for the first location that isn't stay after (not including) (i-1)
                int j=i;
                while(j<pathlength && path[j]==STAYPATH ) j++;
                //So far we have (ZXYYY followed by zero or more stays then something)
                //If the something is YYYYY and we still have any space left before the end...
                if(path[j]==repeat5 && j<pathlength-1)
                {
                    //Hunt for the first location that isn't stay or repeat5: this is the end of the run
                    int e=j+1;
                    while( e<pathlength && (path[e]==-1 || path[e]==repeat5) )
                        e++;
                    (*runstarts)[runcount]=j;//Location of start of run: note this is j not i
                    (*runlengths)[runcount]=e-j;
                    (*runbases)[runcount]=base;
                    runcount++;
                }
            }
        }
    }
    //We allocated the arrays at the start using far too much space: reallocate to what we need
    *runstarts  = (int  *)realloc(*runstarts, runcount*sizeof(int));
    *runlengths = (int  *)realloc(*runlengths,runcount*sizeof(int));
    *runbases   = (int  *)realloc(*runbases,  runcount*sizeof(int));
    return runcount;
}

/*Print some stuff to see what the inputs to homopolymer.c are, for debugging purposes*/
int debugPrint(FILE *stream,scrappie_matrix post, int *viterbipath)
{
    fprintf(stream,"Viterbi path:\n");
    for(int ii=0;ii<500;ii++)
        fprintf(stream,"%d ",(int)(viterbipath[ii]));
    fprintf(stream,"\n");
    //post is a scrappie_matrix
    fprintf(stream,"post->stride = %d \n",post->stride);
    fprintf(stream,"nblock = post->nc = %d \n",post->nc);
    fprintf(stream,"First and last elements of first few rows of post (each ends with stay prob, followed by normalisation check):\n");
    for(int ii=0;ii<21;ii++)
        {
        for(int jj=0;jj<5;jj++)
            {
            fprintf(stream,"%.1f ",post->data.f[ii*post->stride+jj] );
            }
        fprintf(stream,"....");
        for(int jj=STAYPOST-4;jj<=STAYPOST;jj++)
            {
            fprintf(stream,"%.1f ",post->data.f[ii*post->stride + jj ] );
            }
        double sump=0.0;
        for(int jj=0;jj<=STAYPOST;jj++)
            sump=sump+exp(post->data.f[ii*post->stride + jj ]);
        fprintf(stream,": %f\n",sump);
        }
    return 0;
}

/*Print some stuff to the terminal (or whatever stream we're given)
about a particular homopolymer run - used in debugging*/
int printRun(FILE *stream,int *path,scrappie_matrix posteriors,int start,int runstate)
{
    int pstart = start-2;
    int pend = start+10;
    for(int n=pstart;n<pend;n++)
        fprintf(stream,"%d\t",path[n]);
    fprintf(stream,"\n");
    for(int n=pstart;n<pend;n++)
        if(path[n]==STAYPATH)
            fprintf(stream,"S\t");
        else
            fprintf(stream,"%c\t",BASES[(path[n]%64)/16] );
    fprintf(stream,"\n");
    for(int n=pstart;n<pend;n++)
        if(path[n]==STAYPATH)
            fprintf(stream,"S\t");
        else
        {
            int s = path[n];
            fprintf(stream,"%c%c%c%c%c\t",basefromint(s,4),basefromint(s,3),basefromint(s,2),basefromint(s,1),basefromint(s,0) );
        }
    fprintf(stream,"\nS");//Stay probabilities (not normalised)
    for(int n=pstart;n<pend;n++)
        fprintf(stream,"%.2f\t",expf(POSTSHIFT(posteriors,n,STAYPOST)));
    fprintf(stream,"\nR");//Repeat block probabilities (not normalised)
    for(int n=pstart;n<pend;n++)
        fprintf(stream,"%.2f\t",expf(POSTSHIFT(posteriors,n,runstate)));
        
    fprintf(stream,"\n");
    return 0;
}

/*Go through a posterior matrix at the given time, printing all elements whose probabilities are
larger than the given threshold. Used for debugging*/
int printSignificantPosts(FILE * stream,scrappie_matrix post,double threshold,int timeloc,double checkTemp)
{
    double sump=0.0;
    double fullsump=0.0;
    double pstay = exp(post->data.f[ timeloc * post->stride + STAYPOST]);
    for(int j=0;j<=STAYPOST;j++)
    {
        double p = exp(post->data.f[ timeloc * post->stride + j]  );
        if(p>threshold)
        {
            fprintf(stream,"%d:%.4f (p/ps)^1/%.2f=%.4f  ",j,p,checkTemp,pow(p/pstay,1.0/checkTemp));
            sump=sump+p;
        }
        fullsump=fullsump+p;
    }
    fprintf(stream,"[sum=%.4f,fullsum=%.4f]\n",sump,fullsump);
    return 0;
}


/*Do a temperature change according to the recipe suggested by
  Guo et al ('On calibration of modern neural networks', arXiv:1706.04599 (2017))
  For each location we calculate (running over states k)
  p'[k] = p[k]^(1/T) / sum_j p[j]^(1/T)
  Since the posterior matrix is stored in log form q = ln(p) this is
  q'[k] = q[k] /T -ln sum_j exp(q[j]/T)  */
#define PP(posterior,t,i) ( posterior->data.f[ ((t)*(posterior->stride)) + (i) ]  )
int change_temperature(double temperature,scrappie_matrix post)
{
    const int nblock = post->nc;//Number of locations in read
    #ifdef DEBUG
    fprintf(stderr,"Doing temperature change with temp=%.4f\n",temperature);
    double thresh = 0.01;
    int debugloc=nblock/2;
    fprintf(stderr,"Posts at time %d > %.3f before T change:\n",debugloc,thresh);
    printSignificantPosts(stderr,post,thresh,debugloc,temperature);
    #endif
    for(int t=0;t<nblock;t++)
    {
        double sump=0.0;
        for(int k=0;k<=STAYPOST;k++)
        {
            PP(post,t,k)=PP(post,t,k)/temperature;
            sump = sump + exp(PP(post,t,k));
        }
        double logz = log(sump);
        for(int k=0;k<=STAYPOST;k++)
            PP(post,t,k)=PP(post,t,k)-logz;
    }
    #ifdef DEBUG
    fprintf(stderr,"Posts at time %d > %.3f after T change:\n",debugloc,thresh);
    printSignificantPosts(stderr,post,thresh,debugloc,1.0);
    #endif
    return 0;
}
#undef PP

/*Hunt for homopolymer runs, calculate the mean run length using the posterior matrix,
modify the given path so that instead of the Viterbi run length it has the mean
run length.
The array viterbipath is modified by this function so that after a call, it contains
the corrected path (with mean run lengths rather than Viterbi ones).
If pathCalculationFlag is set to HOMOPOLYMER_NOCHANGE (=zero) we do no modifications.
If pathCalculationFlag is HOMOPOLYMER_MEAN (=1) we do the mean calculation.
Future changes may allow other modes of operation, determined by this flag.*/
int homopolymer_path(scrappie_matrix post, int *viterbipath, int pathCalculationFlag)
{
    if(pathCalculationFlag == HOMOPOLYMER_NOCHANGE)
    {
        #ifdef DEBUG
        fprintf(stderr,"No calculation in homopolymer.c\n");
        #endif
        return 0;
    }
    if(pathCalculationFlag != HOMOPOLYMER_MEAN)
    {
        #ifdef DEBUG
        fprintf(stderr,"homopolymer path calculations (homopolymer.c): only HOMOPOLYMER_MEAN implemented so far\n");
        #endif
        return 1;
    }
    #ifdef DEBUG
    fprintf(stderr,"Doing mean homopolymer path calculation in homopolymer.c\n");
    #endif

    #ifdef DEBUG
    debugPrint(stderr,post,viterbipath);//Print some stuff for debugging
    #endif
    const int nblock = post->nc;//Number of locations in read
    int *runstarts;   //Location of start (first ambiguous location) in each homopol run
    int *runlengths;  //Length of ambiguous section
    int *runbases;    //Integer 0-3 representing the base which repeats in the homopol run
    //Find homopolymer runs
    int runcount = findRuns(viterbipath,&runstarts,&runlengths,&runbases,nblock);
    
    for(int nrun=0;nrun<runcount;nrun++)//For each homopolymer run...
        {
            //Calculate Viterbi (as a check against the existing sequence) and mean numbers of non-stays
            //While we're at it, count number of non-stays in the existing sequence
            int nviterbi = 0;
            int ncalcviterbi=0;
            double nmean = 0.0;
            int runstate = runbases[nrun]*341;  //index of a fivemer with the repeat base repeated 5x
            int runlength = runlengths[nrun];
            int ambigfrom = runstarts[nrun];    //location of the first ambiguous block in the homopol run
            int ambigto   = ambigfrom + runlength -1; //location of the last ambiguous block
            //Calculate normalised stay probabilities for each of the ambiguous locations
            for(int i=ambigfrom;i<=ambigto;i++)
            {
                double psu = expf(POSTSHIFT(post,i,STAYPOST));  //Stay probability (un-normalised) - note posts are logged in Scrappie and shifted one step
                double pru = expf(POSTSHIFT(post,i,runstate));  //Repeat block probability (un-normalised)
                double pr = pru / (pru + psu);                         //Normalised repeat block probability
                nmean = nmean + pr;
                if(pr>0.5) ncalcviterbi = ncalcviterbi + 1;
                if(viterbipath[i]==runstate) nviterbi = nviterbi + 1;
            }
            int newn = (int)(nmean+0.5); //nmean is a float so need to round
            #ifdef DEBUG
            //The count of repeat blocks in the Viterbi path should agree with the Viterbi
            //calculation in this function. If not, then output a warning.
            //Sometimes there is disagreement because probabilities differ by
            //tiny amounts - but lots of error messages like this indicate trouble
            if(ncalcviterbi!=nviterbi)
                fprintf(stderr,"************Run %d has viterbi ambig count = %d , calc viterbi ambig count = %d\n",nrun,nviterbi,ncalcviterbi);
            fprintf(stderr,"Run %d : start %d  len %d base %c\n",nrun,ambigfrom,runlength,BASES[runbases[nrun]]);
            fprintf(stderr,"Viterbi nonstays %d, calc vit nonstays %d, mean nonstays %f\n",nviterbi,ncalcviterbi,nmean);
            printRun(stderr,viterbipath,post,ambigfrom,runstate);
            #endif
            //Make modification to the path if necessary
            if(newn!=nviterbi)
            {
                for(int i=0;i<=ambigto-ambigfrom;i++)
                {
                    //Fill in the right number of repeat blocks, putting stays for the rest
                    //(order doesn't matter since this path will be collapsed to a sequence)
                    if(i<newn)
                    {
                        viterbipath[i+ambigfrom]=runstate; 
                        #ifdef DEBUG
                        fprintf(stderr,"i=%d->%d;",i,runstate);
                        #endif
                    }
                    else viterbipath[i+ambigfrom]=STAYPATH;
                }
            #ifdef DEBUG
            fprintf(stderr,"**********************DIFFERENCE***********************\n");
            printRun(stderr,viterbipath,post,ambigfrom,runstate);
            #endif
            }
        #ifdef DEBUG
        fprintf(stderr,"\n");
        #endif
        }
    free(runstarts);
    free(runlengths);
    free(runbases);
    return 0;
}