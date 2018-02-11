#define h102_cxx
#include "h102.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <valarray>

void h102::Loop()
{
//   In a ROOT session, you can do:
//      root> .L h102.C
//      root> h102 t
//      root> t.GetEntry(12); // Fill t data members with entry number 12
//      root> t.Show();       // Show values of entry 12
//      root> t.Show(16);     // Read and show values of entry 16
//      root> t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();
   /*
   int noofcryst = 162;//mod
   
   //Creating and opening data files
   FILE * dataFile;//mod
   dataFile = fopen("xb_data_gamma_isotropic_0.1-10_100000.txt","w");//mod (detector output data)
   FILE * gunTFile;//mod
   gunTFile = fopen("gunTVals_0.1-10.txt","w");//mod("correct" gun energies)
   */
   FILE * depEFile;//mod
   depEFile = fopen("sum_of_dep_energies.txt","w");

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) { //Looping over events
     
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;

      std::valarray<Float_t> energyArray(XBe,XBn);//mod
      fprintf(depEFile,"%f \n",energyArray.sum());//mod

      
      /*
      fprintf(gunTFile,"%f \n",gunT);//mod (writes gun T data to file)
      
      int j = 0;//mod
      
      for (int i=1;i<=noofcryst;i++){
	
	UInt_t * index;//mod
	index = std::find(XBi,XBi+XBn,i);//mod
	if (index != XBi+XBn){
	  fprintf(dataFile,"%f ",XBe[j]); //mod (writes dep. energy in each detector to file)
	  j++;
	} else {
	  fprintf(dataFile,"%f ",0.0);//mod
	}
      }
      
        fprintf(dataFile,"\n");//mod
      */
      printf("%s %lld\n","Iteration: ",jentry);
   }
   //Closing files
   fclose(depEFile);
   /*
   fclose(dataFile);//mod
   fclose(gunTFile);//mod
   */
}
