//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Jun  8 15:59:23 2017 by ROOT version 6.08/06
// from TChain data/
//////////////////////////////////////////////////////////

#ifndef getIncorrectEvents_h
#define getIncorrectEvents_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

// Headers needed by this particular selector


class getIncorrectEvents : public TSelector {
public :
   TTreeReader     fReader;  //!the tree reader
   TTree          *fChain = 0;   //!pointer to the analyzed TTree or TChain

   // Readers to access the data (delete the ones you do not need).
   TTreeReaderValue<Int_t> runNumber = {fReader, "runNumber"};
   TTreeReaderValue<Int_t> eventNumber = {fReader, "eventNumber"};
   TTreeReaderValue<Int_t> MJD = {fReader, "MJD"};
   TTreeReaderValue<Double_t> Time = {fReader, "Time"};
   TTreeReaderArray<Double_t> TelElevation = {fReader, "TelElevation"};
   TTreeReaderArray<Double_t> TelAzimuth = {fReader, "TelAzimuth"};
   TTreeReaderValue<Float_t> ArrayPointing_Elevation = {fReader, "ArrayPointing_Elevation"};
   TTreeReaderValue<Float_t> ArrayPointing_Azimuth = {fReader, "ArrayPointing_Azimuth"};
   TTreeReaderValue<Double_t> WobbleN = {fReader, "WobbleN"};
   TTreeReaderValue<Double_t> WobbleE = {fReader, "WobbleE"};
   TTreeReaderValue<Int_t> MCprimary = {fReader, "MCprimary"};
   TTreeReaderValue<Double_t> MCe0 = {fReader, "MCe0"};
   TTreeReaderValue<Double_t> MCxcore = {fReader, "MCxcore"};
   TTreeReaderValue<Double_t> MCycore = {fReader, "MCycore"};
   TTreeReaderValue<Double_t> MCaz = {fReader, "MCaz"};
   TTreeReaderValue<Double_t> MCze = {fReader, "MCze"};
   TTreeReaderValue<Double_t> MCxoff = {fReader, "MCxoff"};
   TTreeReaderValue<Double_t> MCyoff = {fReader, "MCyoff"};
   TTreeReaderValue<Int_t> MCCorsikaRunID = {fReader, "MCCorsikaRunID"};
   TTreeReaderValue<Int_t> MCCorsikaShowerID = {fReader, "MCCorsikaShowerID"};
   TTreeReaderValue<Float_t> MCFirstInteractionHeight = {fReader, "MCFirstInteractionHeight"};
   TTreeReaderValue<Float_t> MCFirstInteractionDepth = {fReader, "MCFirstInteractionDepth"};
   TTreeReaderValue<ULong64_t> LTrig = {fReader, "LTrig"};
   TTreeReaderValue<UInt_t> NTrig = {fReader, "NTrig"};
   TTreeReaderValue<Int_t> NImages = {fReader, "NImages"};
   TTreeReaderValue<ULong64_t> ImgSel = {fReader, "ImgSel"};
   TTreeReaderArray<UInt_t> ImgSel_list = {fReader, "ImgSel_list"};
   TTreeReaderValue<Int_t> NTtype = {fReader, "NTtype"};
   TTreeReaderArray<ULong64_t> TtypeID = {fReader, "TtypeID"};
   TTreeReaderArray<UInt_t> NImages_Ttype = {fReader, "NImages_Ttype"};
   TTreeReaderValue<Double_t> img2_ang = {fReader, "img2_ang"};
   TTreeReaderValue<Int_t> RecID = {fReader, "RecID"};
   TTreeReaderValue<Double_t> Ze = {fReader, "Ze"};
   TTreeReaderValue<Double_t> Az = {fReader, "Az"};
   TTreeReaderValue<Double_t> Xoff = {fReader, "Xoff"};
   TTreeReaderValue<Double_t> Yoff = {fReader, "Yoff"};
   TTreeReaderValue<Double_t> Xoff_derot = {fReader, "Xoff_derot"};
   TTreeReaderValue<Double_t> Yoff_derot = {fReader, "Yoff_derot"};
   TTreeReaderValue<Double_t> Xcore = {fReader, "Xcore"};
   TTreeReaderValue<Double_t> Ycore = {fReader, "Ycore"};
   TTreeReaderValue<Double_t> stdP = {fReader, "stdP"};
   TTreeReaderValue<Double_t> Chi2 = {fReader, "Chi2"};
   TTreeReaderValue<Float_t> meanPedvar_Image = {fReader, "meanPedvar_Image"};
   TTreeReaderValue<Double_t> DispDiff = {fReader, "DispDiff"};
   TTreeReaderArray<Double_t> R = {fReader, "R"};
   TTreeReaderArray<Double_t> ES = {fReader, "ES"};
   TTreeReaderValue<Double_t> MSCW = {fReader, "MSCW"};
   TTreeReaderValue<Double_t> MSCL = {fReader, "MSCL"};
   TTreeReaderValue<Float_t> MWR = {fReader, "MWR"};
   TTreeReaderValue<Float_t> MLR = {fReader, "MLR"};
   TTreeReaderValue<Double_t> ErecS = {fReader, "ErecS"};
   TTreeReaderValue<Double_t> EChi2S = {fReader, "EChi2S"};
   TTreeReaderValue<Double_t> dES = {fReader, "dES"};
   TTreeReaderValue<Float_t> EmissionHeight = {fReader, "EmissionHeight"};
   TTreeReaderValue<Float_t> EmissionHeightChi2 = {fReader, "EmissionHeightChi2"};
   TTreeReaderValue<UInt_t> NTelPairs = {fReader, "NTelPairs"};
   TTreeReaderValue<Double_t> SizeSecondMax = {fReader, "SizeSecondMax"};


   getIncorrectEvents(TTree * /*tree*/ =0) { }
   virtual ~getIncorrectEvents() { }
   virtual Int_t   Version() const { return 2; }
   virtual void    Begin(TTree *tree);
   virtual void    SlaveBegin(TTree *tree);
   virtual void    Init(TTree *tree);
   virtual Bool_t  Notify();
   virtual Bool_t  Process(Long64_t entry);
   virtual Int_t   GetEntry(Long64_t entry, Int_t getall = 0) { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) { fInput = input; }
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();

   ClassDef(getIncorrectEvents,0);

};

#endif

#ifdef getIncorrectEvents_cxx
void getIncorrectEvents::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the reader is initialized.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   fReader.SetTree(tree);
}

Bool_t getIncorrectEvents::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}


#endif // #ifdef getIncorrectEvents_cxx
