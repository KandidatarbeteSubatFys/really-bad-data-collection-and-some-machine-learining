#define h102_cxx
#include "h102.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <valarray>
#include <string>

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
   dataFile = fopen("crystal_energies.txt","w");//detector output data
   FILE * gunTFile;
   gunTFile = fopen("gun_data.txt","w");//"correct" gun energies
   FILE * depEFile;
   depEFile = fopen("sum_of_dep_energies.txt","w");
   //FILE * myfile; //test file for writing out data
   //myfile = fopen("gamma_testfile.txt","w");

   Long64_t nentries = fChain->GetEntriesFast();

   //Setting up crystal array
   
   const int noofcryst = 140;
   float crystalEnergies[noofcryst] = {0.0};

   Long64_t nbytes = 0, nb = 0;
   
   //Loop over recorded evnets
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
     
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;



      try{
	int test=gunn;
      }
      catch (cling::CompilationException e)
	{
	  gROOT->ProcessLine("std:cout << \"Error: for #gun=1, write --tree=gunlist,FILE when simulating\" <<endl");
	}

      
      for(int i=0;i<gunn;i++){
	//Write gun energies to gun-file
	fprintf(gunTFile,"%f ",gunT[i]);
	//Write cos(theta) to gun-file
	fprintf(gunTFile,"%f ",gunpz[i]/sqrt(gunpz[i]*gunpz[i]+gunpy[i]*gunpy[i]+gunpx[i]*gunpx[i])); // could use pow instead for ^2
	//Write phi to gun-file
	fprintf(gunTFile,"%f ",atan2(gunpy[i],gunpx[i])); // atan2 gives angles from -pi to pi
      }
      fprintf(gunTFile,"\n");
      

      //Write tot dep energy to file
      std::valarray<Float_t> energyArray(DALIe,DALIn);
      fprintf(depEFile,"%f \n",energyArray.sum());

      //Create output data array
      for(int i = 0; i < DALIn; i++){
	crystalEnergies[DALIi[i]-1] = crystalEnergies[DALIi[i]-1] + DALIe[i];
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
   //fclose(myfile);
   
}
