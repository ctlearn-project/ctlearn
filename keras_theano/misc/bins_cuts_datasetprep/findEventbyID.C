#include <TString.h>
#include <TChain.h>
#include <TTree.h>
#include <iostream>
#include <TTreePlayer.h>
#include <string>

void findEventbyID(int eventID,int runNumber)
{

    TString st = "data";

    TChain ch1(st);
    //ch1.Add("/data/nieto/mscw_energy_diffuse/gamma-diffuse*.root");
    ch1.Add("/data/nieto/mscw_energy_diffuse/*.root");

    string selection = string("eventNumber==") + std::to_string(eventID) + string(" && ") + string("runNumber==") + std::to_string(runNumber);
    const char *selection_cstr = selection.c_str();

    ch1.Scan("runNumber:eventNumber:MCe0:ErecS",selection_cstr);

    return;
}

