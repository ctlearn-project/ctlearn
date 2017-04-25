#include <TString.h>
#include <TChain.h>
#include <TTree.h>
#include <iostream>
#include <TTreePlayer.h>

using namespace std;

int generateImageList(TString mypath1, TString mypath2, TString mypath3, TString mypath4)
{

    TString st = "showerpars";

    TChain ch1(st);
    ch1.Add(mypath1);
    ch1.Add(mypath2);
    TChain ch2(st);
    ch2.Add(mypath3);
    ch2.Add(mypath4);

    ((TTreePlayer*)(ch1.GetPlayer()))->SetScanRedirect(true);
    ((TTreePlayer*)(ch1.GetPlayer()))->SetScanFileName("gammalist.txt");
    ch1.Scan("eventNumber","MCe0>1");
    
    ((TTreePlayer*)(ch2.GetPlayer()))->SetScanRedirect(true);
    ((TTreePlayer*)(ch2.GetPlayer()))->SetScanFileName("protonlist.txt");
    ch2.Scan("eventNumber","MCe0>1");
    
    return 0;
}

int main(int argc, char **argv)
{
    //TString mypath1 = "/data/nieto/deeplearning/evn/gamma/20deg/0deg/root";
    TString mypath1 = "/data/nieto/deeplearning/evn/gamma-diffuse/20deg/0deg/*.root";
    TString mypath2 = "/data/nieto/deeplearning/evn/gamma-diffuse/20deg/180deg/*.root";


    //TString mypath2 = "/data/nieto/deeplearning/evn/gamma/20deg/180deg/*root";
    TString mypath3 = "/data/nieto/deeplearning/evn/proton/20deg/0deg/*.root";
    TString mypath4 = "/data/nieto/deeplearning/evn/proton/20deg/180deg/*.root";

    return generateImageList(mypath1,mypath2,mypath3,mypath4);
}
