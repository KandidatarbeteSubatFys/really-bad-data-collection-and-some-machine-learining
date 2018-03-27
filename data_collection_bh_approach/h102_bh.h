//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Mar 27 17:34:42 2018 by ROOT version 6.12/04
// from TTree h102/collect hit tree
// found on file: xb_with_blackhole.root
//////////////////////////////////////////////////////////

#ifndef h102_h
#define h102_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

class h102 {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   UInt_t          eventno;
   UInt_t          seed1;
   UInt_t          seed2;
   UInt_t          gunn;
   Float_t         gunt[1];   //[gunn]
   Float_t         gunx[1];   //[gunn]
   Float_t         guny[1];   //[gunn]
   Float_t         gunz[1];   //[gunn]
   Float_t         gunpx[1];   //[gunn]
   Float_t         gunpy[1];   //[gunn]
   Float_t         gunpz[1];   //[gunn]
   Float_t         gunT[1];   //[gunn]
   Int_t           gunpdg[1];   //[gunn]
   Float_t         XBsumE;
   UInt_t          XBn;
   Float_t         XBt[10];   //[XBn]
   Float_t         XBe[10];   //[XBn]
   UInt_t          XBi[10];   //[XBn]
   Float_t         TSTsumE;
   UInt_t          TSTn;
   Float_t         TSTt[1];   //[TSTn]
   Float_t         TSTT[1];   //[TSTn]
   Float_t         TSTx[1];   //[TSTn]
   Float_t         TSTy[1];   //[TSTn]
   Float_t         TSTz[1];   //[TSTn]
   Float_t         TSTpx[1];   //[TSTn]
   Float_t         TSTpy[1];   //[TSTn]
   Float_t         TSTpz[1];   //[TSTn]
   UInt_t          TSTinn;
   Float_t         TSTint[5];   //[TSTinn]
   Float_t         TSTinx[5];   //[TSTinn]
   Float_t         TSTiny[5];   //[TSTinn]
   Float_t         TSTinz[5];   //[TSTinn]
   Float_t         TSTinpx[5];   //[TSTinn]
   Float_t         TSTinpy[5];   //[TSTinn]
   Float_t         TSTinpz[5];   //[TSTinn]
   Float_t         TSTinT[5];   //[TSTinn]
   Int_t           TSTinpdg[5];   //[TSTinn]

   // List of branches
   TBranch        *b_eventno;   //!
   TBranch        *b_seed1;   //!
   TBranch        *b_seed2;   //!
   TBranch        *b_gunn;   //!
   TBranch        *b_gunt;   //!
   TBranch        *b_gunx;   //!
   TBranch        *b_guny;   //!
   TBranch        *b_gunz;   //!
   TBranch        *b_gunpx;   //!
   TBranch        *b_gunpy;   //!
   TBranch        *b_gunpz;   //!
   TBranch        *b_gunT;   //!
   TBranch        *b_gunpdg;   //!
   TBranch        *b_XBsumE;   //!
   TBranch        *b_XBn;   //!
   TBranch        *b_XBt;   //!
   TBranch        *b_XBe;   //!
   TBranch        *b_XBi;   //!
   TBranch        *b_TSTsumE;   //!
   TBranch        *b_TSTn;   //!
   TBranch        *b_TSTt;   //!
   TBranch        *b_TSTT;   //!
   TBranch        *b_TSTx;   //!
   TBranch        *b_TSTy;   //!
   TBranch        *b_TSTz;   //!
   TBranch        *b_TSTpx;   //!
   TBranch        *b_TSTpy;   //!
   TBranch        *b_TSTpz;   //!
   TBranch        *b_TSTinn;   //!
   TBranch        *b_TSTint;   //!
   TBranch        *b_TSTinx;   //!
   TBranch        *b_TSTiny;   //!
   TBranch        *b_TSTinz;   //!
   TBranch        *b_TSTinpx;   //!
   TBranch        *b_TSTinpy;   //!
   TBranch        *b_TSTinpz;   //!
   TBranch        *b_TSTinT;   //!
   TBranch        *b_TSTinpdg;   //!

   h102(TTree *tree=0);
   virtual ~h102();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef h102_cxx
h102::h102(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("xb_with_blackhole.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("xb_with_blackhole.root");
      }
      f->GetObject("h102",tree);

   }
   Init(tree);
}

h102::~h102()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t h102::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t h102::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void h102::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("eventno", &eventno, &b_eventno);
   fChain->SetBranchAddress("seed1", &seed1, &b_seed1);
   fChain->SetBranchAddress("seed2", &seed2, &b_seed2);
   fChain->SetBranchAddress("gunn", &gunn, &b_gunn);
   fChain->SetBranchAddress("gunt", gunt, &b_gunt);
   fChain->SetBranchAddress("gunx", gunx, &b_gunx);
   fChain->SetBranchAddress("guny", guny, &b_guny);
   fChain->SetBranchAddress("gunz", gunz, &b_gunz);
   fChain->SetBranchAddress("gunpx", gunpx, &b_gunpx);
   fChain->SetBranchAddress("gunpy", gunpy, &b_gunpy);
   fChain->SetBranchAddress("gunpz", gunpz, &b_gunpz);
   fChain->SetBranchAddress("gunT", gunT, &b_gunT);
   fChain->SetBranchAddress("gunpdg", gunpdg, &b_gunpdg);
   fChain->SetBranchAddress("XBsumE", &XBsumE, &b_XBsumE);
   fChain->SetBranchAddress("XBn", &XBn, &b_XBn);
   fChain->SetBranchAddress("XBt", XBt, &b_XBt);
   fChain->SetBranchAddress("XBe", XBe, &b_XBe);
   fChain->SetBranchAddress("XBi", XBi, &b_XBi);
   fChain->SetBranchAddress("TSTsumE", &TSTsumE, &b_TSTsumE);
   fChain->SetBranchAddress("TSTn", &TSTn, &b_TSTn);
   fChain->SetBranchAddress("TSTt", &TSTt, &b_TSTt);
   fChain->SetBranchAddress("TSTT", &TSTT, &b_TSTT);
   fChain->SetBranchAddress("TSTx", &TSTx, &b_TSTx);
   fChain->SetBranchAddress("TSTy", &TSTy, &b_TSTy);
   fChain->SetBranchAddress("TSTz", &TSTz, &b_TSTz);
   fChain->SetBranchAddress("TSTpx", &TSTpx, &b_TSTpx);
   fChain->SetBranchAddress("TSTpy", &TSTpy, &b_TSTpy);
   fChain->SetBranchAddress("TSTpz", &TSTpz, &b_TSTpz);
   fChain->SetBranchAddress("TSTinn", &TSTinn, &b_TSTinn);
   fChain->SetBranchAddress("TSTint", TSTint, &b_TSTint);
   fChain->SetBranchAddress("TSTinx", TSTinx, &b_TSTinx);
   fChain->SetBranchAddress("TSTiny", TSTiny, &b_TSTiny);
   fChain->SetBranchAddress("TSTinz", TSTinz, &b_TSTinz);
   fChain->SetBranchAddress("TSTinpx", TSTinpx, &b_TSTinpx);
   fChain->SetBranchAddress("TSTinpy", TSTinpy, &b_TSTinpy);
   fChain->SetBranchAddress("TSTinpz", TSTinpz, &b_TSTinpz);
   fChain->SetBranchAddress("TSTinT", TSTinT, &b_TSTinT);
   fChain->SetBranchAddress("TSTinpdg", TSTinpdg, &b_TSTinpdg);
   Notify();
}

Bool_t h102::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void h102::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t h102::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef h102_cxx
