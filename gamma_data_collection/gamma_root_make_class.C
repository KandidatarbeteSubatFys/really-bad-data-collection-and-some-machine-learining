
void gamma_root_make_class(const char * file){
  TFile *a=TFile::Open(file);
  TTree *h102=NULL;
  a->GetObject("h102",h102);
  h102->MakeClass();
  
}
