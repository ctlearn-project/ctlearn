#include <TString.h>
#include <TChain.h>
#include <TTree.h>
#include <iostream>
#include <TTreePlayer.h>
#include <string>

using namespace std;

int generateEventList(TString mypath1, string label1, TString mypath2, string label2, string selection)
{

    TString st = "data";

    TChain ch1(st);
    ch1.Add(mypath1);
    TChain ch2(st);
    ch2.Add(mypath2);

    string list = "_list.txt";
    string filename1 = label1 + list;
    string filename2 = label2 + list;

    ((TTreePlayer*)(ch1.GetPlayer()))->SetScanRedirect(true);
    ((TTreePlayer*)(ch1.GetPlayer()))->SetScanFileName(filename1.c_str());
    ch1.Scan("eventNumber",selection.c_str());

    ((TTreePlayer*)(ch2.GetPlayer()))->SetScanRedirect(true);
    ((TTreePlayer*)(ch2.GetPlayer()))->SetScanFileName(filename2.c_str());
    ch2.Scan("eventNumber",selection.c_str());

    return 0;
}
int main(int argc, char **argv)
{
        
    //TString mypath1 = "/data/nieto/deeplearning/evn/gamma/20deg/0deg/root";
    TString mypath1 = "/data/nieto/mscw_energy_diffuse/gamma-diffuse*.root";
    string label1 = "gamma-diffuse";

    //TString mypath2 = "/data/nieto/deeplearning/evn/gamma/20deg/180deg/*root";
    TString mypath2 = "/data/nieto/mscw_energy_diffuse/proton*.root";
    string label2 = "proton";

    //string cut_selection_conditions = "MCe0>10 && MSCW>0.5 && MSCL >0.2 && EChi2S > 10000000 && dES > 0 && EmissionHeight > 0 "; 

    string cut_selection_conditions = "MCe0>5"; 

    return generateEventList(mypath1,label1,mypath2,label2,cut_selection_conditions);
}


