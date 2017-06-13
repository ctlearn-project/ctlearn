#include <TString.h>
#include <TChain.h>
#include <TTree.h>
#include <iostream>
#include <TTreePlayer.h>
#include <string>

void findEventbyID(int eventID)
{

    TString st = "data";

    TChain ch1(st);
    //ch1.Add("/data/nieto/mscw_energy_diffuse/gamma-diffuse*.root");
    ch1.Add("/data/nieto/mscw_energy_diffuse/proton*.root");

    string selection = string("eventNumber==") + std::to_string(eventID);
    const char *selection_cstr = selection.c_str();

    ch1.Scan("eventNumber:MCe0:ErecS",selection_cstr);

    return;
}

