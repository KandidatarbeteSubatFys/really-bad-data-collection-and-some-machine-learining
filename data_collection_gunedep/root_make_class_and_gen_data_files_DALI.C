#include <iostream>
#include <fstream>
using namespace std;

// This code is for DALI only

void root_make_class_and_gen_data_files_DALI(const char * rootFile) {
  // Read root file and make the h102-class
  TFile *a=TFile::Open(rootFile);
  TTree *h102=NULL;
  a->GetObject("h102",h102);
  h102->MakeClass();

  // Overwrite h102.C
  ofstream h102file("h102.C",ios::trunc);
  ifstream h102backupfile("h102_backup_gunedep_DALI.C");
  string line;
  
  if ( h102file.is_open() && h102backupfile.is_open() ){
    while ( getline(h102backupfile,line) ){
      h102file << line << '\n';
    }
    h102file.close();
    h102backupfile.close();
  }
  else printf("%s","Unable to open files...");

  // Generate data files 	
  gROOT->ProcessLine(".L h102.C");
  gROOT->ProcessLine("h102 t");
  gROOT->ProcessLine("t.Loop()");
  
}
