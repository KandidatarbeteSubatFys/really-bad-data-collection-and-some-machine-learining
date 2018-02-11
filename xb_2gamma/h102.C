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

   //Creating and opening data files
   FILE * dataFile;
   dataFile = fopen("xb_data_2gamma_isotropic_0.1-10_100000.txt","w");//detector output data
   FILE * gunTFile;
   gunTFile = fopen("gunTVals_0.1-10.txt","w");//"correct" gun energies
   FILE * depEFile;
   depEFile = fopen("sum_of_dep_energies.txt","w");

   Long64_t nentries = fChain->GetEntriesFast();

   //Setting up crystal array
   const int noofcryst = 162;
   float crystalEnergies[noofcryst] = {0.0};

   Long64_t nbytes = 0, nb = 0;
   
   //Loop over recorded evnets
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
     
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;

      //Write gun energies to file
      fprintf(gunTFile,"%f %f \n",gunT[0],gunT[1]);

      //Write tot dep energy to file
      std::valarray<Float_t> energyArray(XBe,XBn);
      fprintf(depEFile,"%f \n",energyArray.sum());

      //Create output data array
      for(int i = 0; i < XBn; i++){
	crystalEnergies[XBi[i]-1] = crystalEnergies[XBi[i]-1] + XBe[i];
      }
      for(int i = 0; i < noofcryst; i++){
	//Write output data to file
	fprintf(dataFile,"%f ",crystalEnergies[i]);
	crystalEnergies[i] = 0.0;
      }
      fprintf(dataFile,"\n");
      
   }
   fclose(gunTFile);
   fclose(dataFile);
   fclose(depEFile);
   
}
