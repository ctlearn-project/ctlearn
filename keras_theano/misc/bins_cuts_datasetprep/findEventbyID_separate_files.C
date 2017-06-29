#include <TString.h>
#include <TChain.h>
#include <TTree.h>
#include <iostream>
#include <TTreePlayer.h>
#include <string>
#include <list>

void findEventbyID_separate_files(int eventID)
{

    TString st = "data";

    std::list<const char*> file_list = {"/data/nieto/mscw_energy_diffuse/gamma-diffuse.N.3AL0M08-5-S_DL_ID0-LaPalma-1.mscw.root","/data/nieto/mscw_energy_diffuse/gamma-diffuse.N.3AL0M08-5-S_DL_ID0-LaPalma-1001.mscw.root","/data/nieto/mscw_energy_diffuse/gamma-diffuse.N.3AL0M08-5-S_DL_ID0-LaPalma-1501.mscw.root","/data/nieto/mscw_energy_diffuse/gamma-diffuse.N.3AL0M08-5-S_DL_ID0-LaPalma-2001.mscw.root","/data/nieto/mscw_energy_diffuse/gamma-diffuse.N.3AL0M08-5-S_DL_ID0-LaPalma-2501.mscw.root","/data/nieto/mscw_energy_diffuse/gamma-diffuse.N.3AL0M08-5-S_DL_ID0-LaPalma-3001.mscw.root","/data/nieto/mscw_energy_diffuse/gamma-diffuse.N.3AL0M08-5-S_DL_ID0-LaPalma-3501.mscw.root","/data/nieto/mscw_energy_diffuse/gamma-diffuse.N.3AL0M08-5-S_DL_ID0-LaPalma-4001.mscw.root","/data/nieto/mscw_energy_diffuse/gamma-diffuse.N.3AL0M08-5-S_DL_ID0-LaPalma-501.mscw.root"};

    for (const char* i : file_list) 
    {
    
        TFile *f = TFile::Open(i);     

        TTree *t = (TTree *)f->Get(st);

        string selection = string("eventNumber==") + std::to_string(eventID);
        const char *selection_cstr = selection.c_str();

        t->Scan("eventNumber:MCe0:ErecS",selection_cstr);

    }

       return;
}

