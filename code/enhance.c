/*************************************************************************************
			       DEPARTMENT OF ELECTRICAL AND ELECTRONIC ENGINEERING
					   		     IMPERIAL COLLEGE LONDON 

 				      EE 3.19: Real Time Digital Signal Processing
					       Dr Paul Mitcheson and Daniel Harvey

				        		 PROJECT: Frame Processing

 				            ********* ENHANCE. C **********
							 Shell for speech enhancement 

  		Demonstrates overlap-add frame processing (interrupt driven) on the DSK. 

 *************************************************************************************
 				             By Danny Harvey: 21 July 2006
							 Updated for use on CCS v4 Sept 2010
 ************************************************************************************/
/*
 *	You should modify the code so that a speech enhancement project is built 
 *  on top of this template.
 */
/**************************** Pre-processor statements ******************************/
//  library required when using calloc
#include <stdlib.h>
//  Included so program can make use of DSP/BIOS configuration tool.  
#include "dsp_bios_cfg.h"

/* The file dsk6713.h must be included in every program that uses the BSL.  This 
   example also includes dsk6713_aic23.h because it uses the 
   AIC23 codec module (audio interface). */
#include "dsk6713.h"
#include "dsk6713_aic23.h"

// math library (trig functions)
#include <math.h>

/* Some functions to help with Complex algebra and FFT. */
#include "cmplx.h"      
#include "fft_functions.h"  

// Some functions to help with writing/reading the audio ports when using interrupts.
#include <helper_functions_ISR.h>

#define WINCONST 0.85185			/* 0.46/0.54 for Hamming window */
#define FSAMP 8000.0		/* sample frequency, ensure this matches Config for AIC */
#define FFTLEN 256					/* fft length = frame length 256/8000 = 32 ms*/
#define NFREQ (1+FFTLEN/2)			/* number of frequency bins from a real FFT */
#define OVERSAMP 4					/* oversampling ratio (2 or 4) */  
#define FRAMEINC (FFTLEN/OVERSAMP)	/* Frame increment */
#define CIRCBUF (FFTLEN+FRAMEINC)	/* length of I/O buffers */
#define MLEN 1.5/(FFTLEN/(FSAMP*OVERSAMP))				/* length of M buffers, roughly equals to 2.5s */

#define OUTGAIN 16000.0				/* Output gain for DAC */
#define INGAIN  (1.0/16000.0)		/* Input gain for ADC  */
// PI defined here for use in your code 
#define PI 3.141592653589793
#define TFRAME FRAMEINC/FSAMP       /* time between calculation of each frame */


/******************************* Global declarations ********************************/

/* Audio port configuration settings: these values set registers in the AIC23 audio 
   interface to configure it. See TI doc SLWS106D 3-3 to 3-10 for more info. */
DSK6713_AIC23_Config Config = { \
			 /**********************************************************************/
			 /*   REGISTER	            FUNCTION			      SETTINGS         */ 
			 /**********************************************************************/\
    0x0017,  /* 0 LEFTINVOL  Left line input channel volume  0dB                   */\
    0x0017,  /* 1 RIGHTINVOL Right line input channel volume 0dB                   */\
    0x01f9,  /* 2 LEFTHPVOL  Left channel headphone volume   0dB                   */\
    0x01f9,  /* 3 RIGHTHPVOL Right channel headphone volume  0dB                   */\
    0x0011,  /* 4 ANAPATH    Analog audio path control       DAC on, Mic boost 20dB*/\
    0x0000,  /* 5 DIGPATH    Digital audio path control      All Filters off       */\
    0x0000,  /* 6 DPOWERDOWN Power down control              All Hardware on       */\
    0x0043,  /* 7 DIGIF      Digital audio interface format  16 bit                */\
    0x008d,  /* 8 SAMPLERATE Sample rate control        8 KHZ-ensure matches FSAMP */\
    0x0001   /* 9 DIGACT     Digital interface activation    On                    */\
			 /**********************************************************************/
};

// Codec handle:- a variable used to identify audio interface  
DSK6713_AIC23_CodecHandle H_Codec;

float *inbuffer, *outbuffer;   		/* Input/output circular buffers */
float *inframe, *outframe;          /* Input and output frames */
float *inwin, *outwin;              /* Input and output windows */
complex *fftres;
float *M1;
float *M2;
float *M3;
float *M4;
float *mag;
float *noisemin;
float *g;
float *lpf_mag;
float *lpf_power;
float *y1,*y2,*y3;
float *N_over_X;

float alpha = 5;
float alpha_lf = 8;
float lambda = 0.04;
float tao = 0.1;
complex K1;
int mode=3;
int gmode = 0;
int lpfnoise = 1;
int mval = 3;
int gval = 0;
int reducemusical = 0;
int lfthreshold = 8;
int residual_noise_red = 0;
float RNRThres = 8000;
float amp = 1;
float alpha_lowSNR = 5;
float s = 6.667;
int lpfinput=0;
int highf = 110;
float snrthres = 2;




float ingain, outgain;				/* ADC and DAC gains */ 
float cpufrac; 						/* Fraction of CPU time used */
volatile int io_ptr=0;              /* Input/ouput pointer for circular buffers */
volatile int frame_ptr=0;           /* Frame pointer */
volatile int m_ptr=MLEN;           /* m pointer */


 /******************************* Function prototypes *******************************/
void init_hardware(void);    	/* Initialize codec */ 
void init_HWI(void);            /* Initialize hardware interrupts */
void ISR_AIC(void);             /* Interrupt service routine for codec */
void process_frame(void);       /* Frame processing routine */
void noise_estimation (float *magnitude);
void init_noisebuff (void) ;
float calculate_g (float el1, float el2);
float min_3 (float el1,float el2, float el3);
           
/********************************** Main routine ************************************/
void main()
{      

  	int k; // used in various for loops
  
/*  Initialize and zero fill arrays */  

	inbuffer	= (float *) calloc(CIRCBUF, sizeof(float));	/* Input array */
    outbuffer	= (float *) calloc(CIRCBUF, sizeof(float));	/* Output array */
	inframe		= (float *) calloc(FFTLEN, sizeof(float));	/* Array for processing*/
    outframe	= (float *) calloc(FFTLEN, sizeof(float));	/* Array for processing*/
    inwin		= (float *) calloc(FFTLEN, sizeof(float));	/* Input window */
    outwin		= (float *) calloc(FFTLEN, sizeof(float));	/* Output window */
    fftres		= (complex *) calloc(FFTLEN, sizeof(complex));	/* FFT result */
    
    M1			= (float *) calloc(FFTLEN, sizeof(float));	/* M1 */
    M2			= (float *) calloc(FFTLEN, sizeof(float));	/* M2 */
    M3			= (float *) calloc(FFTLEN, sizeof(float));	/* M3 */
    M4			= (float *) calloc(FFTLEN, sizeof(float));	/* M4 */
    mag			= (float *) calloc(FFTLEN, sizeof(float));	/* mag */
    

    lpf_mag		= (float *) calloc(FFTLEN, sizeof(float));	/* lpf_mag */
	lpf_power	= (float *) calloc(FFTLEN, sizeof(float));	/* lpf_power */
    noisemin	= (float *) calloc(FFTLEN, sizeof(float));	/* Noise Estimation */
    g			= (float *) calloc(FFTLEN, sizeof(float));	/* g(w) */
    
    y1			= (float *) calloc(FFTLEN, sizeof(float));	/* y1 */
    y2			= (float *) calloc(FFTLEN, sizeof(float));	/* y2 */
    y3			= (float *) calloc(FFTLEN, sizeof(float));	/* y3 */
	N_over_X	= (float *) calloc(FFTLEN, sizeof(float));	/* y3 */

	
	/* initialize board and the audio port */
  	init_hardware();
  
  	/* initialize hardware interrupts */
  	init_HWI();    
  	init_noisebuff ();    
  	
  
/* initialize algorithm constants */  
                       
  	for (k=0;k<FFTLEN;k++)
	{                           
	inwin[k] = sqrt((1.0-WINCONST*cos(PI*(2*k+1)/FFTLEN))/OVERSAMP);
	outwin[k] = inwin[k]; 
	} 
  	ingain=INGAIN;
  	outgain=OUTGAIN;
  	K1 = cexp(cmplx(-TFRAME/tao,0)); // change tao to tau // defined to be complex?

  	    

 							
  	/* main loop, wait for interrupt */  
  	while(1) 	process_frame();
}
    
/********************************** init_hardware() *********************************/  
void init_hardware()
{
    // Initialize the board support library, must be called first 
    DSK6713_init();
    
    // Start the AIC23 codec using the settings defined above in config 
    H_Codec = DSK6713_AIC23_openCodec(0, &Config);

	/* Function below sets the number of bits in word used by MSBSP (serial port) for 
	receives from AIC23 (audio port). We are using a 32 bit packet containing two 
	16 bit numbers hence 32BIT is set for  receive */
	MCBSP_FSETS(RCR1, RWDLEN1, 32BIT);	

	/* Configures interrupt to activate on each consecutive available 32 bits 
	from Audio port hence an interrupt is generated for each L & R sample pair */	
	MCBSP_FSETS(SPCR1, RINTM, FRM);

	/* These commands do the same thing as above but applied to data transfers to the 
	audio port */
	MCBSP_FSETS(XCR1, XWDLEN1, 32BIT);	
	MCBSP_FSETS(SPCR1, XINTM, FRM);	
	

}
/********************************** init_HWI() **************************************/ 
void init_HWI(void)
{
	IRQ_globalDisable();			// Globally disables interrupts
	IRQ_nmiEnable();				// Enables the NMI interrupt (used by the debugger)
	IRQ_map(IRQ_EVT_RINT1,4);		// Maps an event to a physical interrupt
	IRQ_enable(IRQ_EVT_RINT1);		// Enables the event
	IRQ_globalEnable();				// Globally enables interrupts

}
        
/******************************** process_frame() ***********************************/  
void process_frame(void)
{
	int k, m; 
	int io_ptr0; 
	float gtemp;
	float el1;

	float* p;

	/* work out fraction of available CPU time used by algorithm */    
	cpufrac = ((float) (io_ptr & (FRAMEINC - 1)))/FRAMEINC;  
		
	/* wait until io_ptr is at the start of the current frame */ 	
	while((io_ptr/FRAMEINC) != frame_ptr); 
	
	/* then increment the framecount (wrapping if required) */ 
	if (++frame_ptr >= (CIRCBUF/FRAMEINC)) frame_ptr=0;
 	
 	/* save a pointer to the position in the I/O buffers (inbuffer/outbuffer) where the 
 	data should be read (inbuffer) and saved (outbuffer) for the purpose of processing */
 	io_ptr0=frame_ptr * FRAMEINC;
	
	/* copy input data from inbuffer into inframe (starting from the pointer position) */ 
	 
	m=io_ptr0;
    for (k=0;k<FFTLEN;k++)
	{                           
		inframe[k] = inbuffer[m] * inwin[k]; 
		if (++m >= CIRCBUF) m=0; /* wrap if required */
	} 
	
	/************************* DO PROCESSING OF FRAME  HERE **************************/
	
	
	/* please add your code, at the moment the code simply copies the input to the 
	ouptut with no processing */	 
	if (mval != mode) {
		init_noisebuff();
		mval = mode;
	}
	if (gval != gmode){
		init_noisebuff();
		gval = gmode;
	}						      	
										
    for (k=0;k<FFTLEN;k++)
	{                           
		fftres[k] = cmplx (inframe[k],0);/* copy input into fftres */ 
	} 
	
	fft (FFTLEN, fftres);
	
	if (lpfinput == 1){
		for (k=0;k<FFTLEN; k++){
			if (k>highf && k <FFTLEN-highf-1) fftres[k] = cmplx(0,0);
		}
	}
	
	for (k=0;k<FFTLEN; k++){
		mag[k] = cabs(fftres[k]);
	}

	
	//frequency domain processing
	if (mode == 0){ 
		// do nothing
	} 
	else {
		// noise estimation
		if(mode==1){
			noise_estimation(mag);
		} else if (mode==2){
			for (k=0;k<FFTLEN;k++){
				lpf_mag[k] = (1-K1.r)*mag[k] + K1.r*lpf_mag[k] ;
			}
			noise_estimation(lpf_mag);	
		} else if (mode==3){
			for (k=0;k<FFTLEN;k++){
				lpf_power[k] = (1-K1.r)*mag[k]*mag[k] + K1.r*lpf_power[k];
				lpf_mag[k] = sqrt(lpf_power[k]) ;
			}
			
			noise_estimation(lpf_mag);
		}
		
		// noise subtraction
		if (gmode == 0){
			for (k=0;k<FFTLEN;k++){
				gtemp = 1-(noisemin[k]/mag[k]);           
				g[k] = calculate_g(lambda,gtemp); // use max(a,b)
			}
		} else if (gmode==1){
			for (k=0;k<FFTLEN;k++){
				el1 = lambda*noisemin[k]/mag[k];
				gtemp = 1-(noisemin[k]/mag[k]);
				g[k] = calculate_g(el1,gtemp);
			}
		} else if (gmode==2){
			for (k=0;k<FFTLEN;k++){
				el1 = lambda*lpf_mag[k]/mag[k];
				gtemp = 1-(noisemin[k]/mag[k]);
				g[k] = calculate_g(el1,gtemp);
			}
		} else if (gmode==3){
			for (k=0;k<FFTLEN;k++){
				el1 = lambda*noisemin[k]/lpf_mag[k];
				gtemp = 1-(noisemin[k]/lpf_mag[k]);
				g[k] = calculate_g(el1,gtemp);
			}
		} else if (gmode == 4){
			for (k=0;k<FFTLEN;k++){
				el1 = lambda;
				gtemp = 1-(noisemin[k]/lpf_mag[k]);
				g[k] = calculate_g(el1,gtemp);
			}
		}else if (gmode==5){
			for (k=0;k<FFTLEN;k++){
				el1 = lambda;
				gtemp = sqrt(1-(noisemin[k]/mag[k])*(noisemin[k]/mag[k]));
				g[k] = calculate_g(el1,gtemp);
			}
		}else if (gmode==6){
			for (k=0;k<FFTLEN;k++){
				el1 = lambda*noisemin[k]/mag[k];
				gtemp = sqrt(1-(noisemin[k]/mag[k])*(noisemin[k]/mag[k]));
				g[k] = calculate_g(el1,gtemp);
			}
		} else if (gmode==7){
			for (k=0;k<FFTLEN;k++){
				el1 = lambda*lpf_mag[k]/mag[k];
				gtemp = sqrt(1-(noisemin[k]/mag[k])*(noisemin[k]/mag[k]));
				g[k] = calculate_g(el1,gtemp);
			}
		} else if (gmode==8){
			for (k=0;k<FFTLEN;k++){
				el1 = lambda*noisemin[k]/lpf_mag[k];
				gtemp = sqrt(1-(noisemin[k]/lpf_mag[k])*(noisemin[k]/lpf_mag[k]));
				g[k] = calculate_g(el1,gtemp);
			}
		} else if (gmode == 9){
			for (k=0;k<FFTLEN;k++){
				el1 = lambda;
				gtemp = sqrt(1-(noisemin[k]/lpf_mag[k])*(noisemin[k]/lpf_mag[k]));
				g[k] = calculate_g(el1,gtemp);
			}
		}
		
//		for (k=0;k<FFTLEN;k++){
//				           
//				if ((mag[k]/noisemin[k])>snrthres) g[k] = 0; // use max(a,b)
//			}
//		
		for (k=0;k<FFTLEN;k++)
		{              
			fftres[k] = rmul(g[k],fftres[k]);
		}
	}
	
	ifft (FFTLEN, fftres);
	
	
	//may need to convert output from ifft from complex to real
	for (k=0;k<FFTLEN; k++){
		y3[k] = amp*fftres[k].r;
	}
	
	if (residual_noise_red == 0){
		outframe = y3;
	} else if (residual_noise_red == 1) {
		for (k=0;k<FFTLEN; k++){
			if (N_over_X[k]>RNRThres){ 
				outframe[k] = min_3(y1[k],y2[k],y3[k]);
			}else{
				outframe[k] = y2[k];
			}
//			outframe[k] = y2[k];
			N_over_X[k]=(noisemin[k]/mag[k]);
		}
		p=y1;
		y1=y2;
		y2=y3;
		y3=p;
	}
		
	
	/********************************************************************************/
	
    /* multiply outframe by output window and overlap-add into output buffer */  
                           
	m=io_ptr0;
    
    for (k=0;k<(FFTLEN-FRAMEINC);k++) 
	{    										/* this loop adds into outbuffer */                       
	  	outbuffer[m] = outbuffer[m]+outframe[k]*outwin[k];   
		if (++m >= CIRCBUF) m=0; /* wrap if required */
	}         
    for (;k<FFTLEN;k++) 
	{                           
		outbuffer[m] = outframe[k]*outwin[k];   /* this loop over-writes outbuffer */        
	    m++;
	}	                                   
}

void noise_estimation (float *magnitude){
	//M1 always points to the location with the latest noise information
	int k;
	float* tempptr;
	float SNRTemp;
	if (m_ptr >= MLEN){ // m_ptr >= MLEN?
		for (k=0;k<FFTLEN;k++)
		{              
			noisemin[k] = M1[k];             
			if(M2[k]<noisemin[k]) noisemin[k] = M2[k];
			if(M3[k]<noisemin[k]) noisemin[k] = M3[k];
			if(M4[k]<noisemin[k]) noisemin[k] = M4[k];
			if (reducemusical == 0){ //multiply minimum noise spectrum with scaling factor
				noisemin[k] = alpha * noisemin[k];
			} else if (reducemusical == 1){
				if (k <lfthreshold) {
					noisemin[k] = alpha_lf*noisemin[k];
				} else if (k>FFTLEN-lfthreshold-1){
					noisemin[k] = alpha_lf*noisemin[k];
				} else{
					noisemin[k] = alpha*noisemin[k];
				}
			
			}else if (reducemusical == 2){
				SNRTemp = 20*log10(lpf_mag[k])/(noisemin[k]);
				//SNRTemp = lpf_mag[k]/noisemin[k];
				if (SNRTemp>20){
					noisemin[k]=noisemin[k]* (alpha-20/s);
				} else if (SNRTemp <-5){
					noisemin[k] = noisemin[k]*(alpha+5/s);
				} else{
					noisemin[k] = noisemin[k] * (alpha-SNRTemp/s);
				}
			}
		}
		
		if (lpfnoise == 1){
			for (k=0;k<FFTLEN;k++)
				{
					noisemin[k] = (1-K1.r)*noisemin[k] + K1.r*noisemin[k] ;
				}
		}
		
		m_ptr = 0;
		tempptr = M1; // tempptr holds the location currently pointed by M1
		M1=M4; //M1 points to the location previously pointed by M4
		M4=M3; //M4 points to the location previously pointed by M3
		M3=M2; //M3 points to the location previously pointed by M2
		M2=tempptr; //M2 points to the location previously pointed by M1
		for (k=0;k<FFTLEN; k++){
			M1[k] = magnitude[k];
		}
	} else{
		
		for (k=0;k<FFTLEN;k++)
		{                           
			if(magnitude[k]<M1[k]) M1[k] = magnitude[k];
		}
		m_ptr+=1;
	}

	}

float calculate_g (float el1, float el2){
	if (el2>el1){
		return el2;
	} else{
		return el1;
	}	
}

void init_noisebuff (void)  {
	int k;
	for (k=0;k<FFTLEN;k++){
		M1[k] = 20000;
		M2[k] = 20000;
		M3[k] = 20000;
		M4[k] = 20000;
		noisemin[k] = 20000;
		y1[k] = FLT_MAX;
		y2[k] = FLT_MAX;
		y3[k] = FLT_MAX;
	}
}
	
float min_3 (float el1,float el2, float el3){
	if( (el1 <= el2) && (el1 <= el3)) return el1;
	if( (el2 <= el1) && (el2 <= el3)) return el2;
	if( (el3 <= el1) && (el3 <= el2)) return el3;
	return el2; //default
}
			
		
        
/*************************** INTERRUPT SERVICE ROUTINE  *****************************/

// Map this to the appropriate interrupt in the CDB file
   
void ISR_AIC(void)
{       
	short sample;
	/* Read and write the ADC and DAC using inbuffer and outbuffer */
	
	sample = mono_read_16Bit();
	inbuffer[io_ptr] = ((float)sample)*ingain;
		/* write new output data */
	mono_write_16Bit((int)(outbuffer[io_ptr]*outgain)); 
	
	/* update io_ptr and check for buffer wraparound */    
	
	if (++io_ptr >= CIRCBUF) io_ptr=0;
}

/************************************************************************************/
