#include <TH1F.h>
#include <TString.h>
#include <TStyle.h>
#include <TChain.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TLine.h>
#include <TLegend.h>

#include <iostream>

using namespace std;

bool plotHistos(TH1F *h1, TH1F *h2, bool bnorm, bool bsave, bool blog, TString label, TString label1, TString label2, TString tag,bool debug);

int plotImgPars(TString mypath1, TString label1, TString mypath2, TString label2, TString tag,int teltoana)
{
  //int plotImgPars(TString mypath1 = "/Users/nieto/tmp/plotImgPars/mpik/*root", TString mypath2 = "/Users/nieto/tmp/plotImgPars/gtech/*root"){

  gStyle->SetTitleFontSize(0.04);
  gStyle->SetTitleFont(43);  
  gStyle->SetTitleBorderSize(0);
  gStyle->SetTitleFillColor(0);

  // 1=LST, 2=MST
  //int ntel1 = 4;
  int ntel1 = 8;
  int ntel2 = 15;

  TString tellabel;
  int ntel_0;
  int ntel_f;
  int nbinspedvar;
  float minpedvar;
  float maxpedvar;
  float minsize;

  if (teltoana == 1)
  {
    tellabel = "LST";
    ntel_0 = 1;
    ntel_f = ntel1;
    nbinspedvar = 1000;
    minpedvar = 140;
    maxpedvar = 220;
    minsize = log10(500);
  }
  else if (teltoana == 2)
  {
    tellabel = "MST-FC";
    ntel_0 = ntel1+1;
    ntel_f = ntel1+ntel2;
    nbinspedvar = 1000;
    minpedvar = 80;
    maxpedvar = 95;
    minsize = log10(100);
  }
  else
  {
      cout << "Wrong telescope type. Exiting..." << endl;
    return -1;
  }
  

  TString st;

  int nbins = 100;// bins in each histogram

  //max and min values for event-wise histograms
  float minMCe0 = 0;
  float maxMCe0 = 5;
  float minMCxcore = -600;
  float maxMCxcore = 800;
  float minMCycore = -600;
  float maxMCycore = 800;
  float minMCxoff = -8;
  float maxMCxoff = 10;
  float minMCyoff = -8;
  float maxMCyoff = 10;

  //max and min values for image-wise histograms
  float maxsize = log10(1e8);
  float minlength = 0;
  float maxlength = 1;
  float minwidth = 0;
  float maxwidth = 0.5;
  float mincen_x = -5;
  float maxcen_x = 5;
  float mincen_y = -5;
  float maxcen_y = 5;
  float mindist = 0;
  float maxdist = 5;
  int minntubes = 0;
  int maxntubes = 500;
  float minasymmetry = -2;
  float maxasymmetry = 2;
  float mintgrad_x = -30;
  float maxtgrad_x = 60;
  float mincosphi = -1;
  float maxcosphi = 1;
  float minsinphi = -1;
  float maxsinphi = 1;

  //create image-wise histograms
  TH1F *sizeaux = new TH1F("sizeaux","log10size",nbins,minsize,maxsize);
  TH1F *size1 = new TH1F("size1","log10size",nbins,minsize,maxsize);
  size1->Sumw2();
  TH1F *size2 = new TH1F("size2","log10(size)",nbins,minsize,maxsize);
  size2->Sumw2();
  TH1F *lengthaux = new TH1F("lengthaux","length",nbins,minlength,maxlength);
  TH1F *length1 = new TH1F("length1","length",nbins,minlength,maxlength);
  length1->Sumw2();
  TH1F *length2 = new TH1F("length2","length",nbins,minlength,maxlength);
  length2->Sumw2();
  TH1F *widthaux = new TH1F("widthaux","width",nbins,minwidth,maxwidth);
  TH1F *width1 = new TH1F("width1","width",nbins,minwidth,maxwidth);
  width1->Sumw2();
  TH1F *width2 = new TH1F("width2","width",nbins,minwidth,maxwidth);
  width2->Sumw2();
  TH1F *pedvaraux = new TH1F("pedvaraux","meanPedvar_Image",nbinspedvar,minpedvar,maxpedvar);
  TH1F *pedvar1 = new TH1F("pedvar1","meanPedvar_Image",nbinspedvar,minpedvar,maxpedvar);
  pedvar1->Sumw2();
  TH1F *pedvar2 = new TH1F("pedvar2","meanPedvar_Image",nbinspedvar,minpedvar,maxpedvar);
  pedvar2->Sumw2();
  TH1F *cen_xaux = new TH1F("cen_xaux","cen_x",nbins,mincen_x,maxcen_x);
  TH1F *cen_yaux = new TH1F("cen_yaux","cen_y",nbins,mincen_y,maxcen_y);
  TH1F *cen_x1 = new TH1F("cen_x1","cen_x",nbins,mincen_x,maxcen_x);
  cen_x1->Sumw2();
  TH1F *cen_x2 = new TH1F("cen_x2","cen_x",nbins,mincen_x,maxcen_x);
  cen_x2->Sumw2();
  TH1F *cen_y1 = new TH1F("cen_y1","cen_y",nbins,mincen_y,maxcen_y);
  cen_y1->Sumw2();
  TH1F *cen_y2 = new TH1F("cen_y2","cen_y",nbins,mincen_y,maxcen_y);
  cen_y2->Sumw2();
  TH1F *distaux = new TH1F("distaux","dist",nbins,mindist,maxdist);
  TH1F *dist1 = new TH1F("dist1","dist",nbins,mindist,maxdist);
  dist1->Sumw2();
  TH1F *dist2 = new TH1F("dist2","dist",nbins,mindist,maxdist);
  dist2->Sumw2();
  TH1F *ntubesaux = new TH1F("ntubesaux","ntubes",nbins,minntubes,maxntubes);
  TH1F *ntubes1 = new TH1F("ntubes1","ntubes",nbins,minntubes,maxntubes);
  ntubes1->Sumw2();
  TH1F *ntubes2 = new TH1F("ntubes2","ntubes",nbins,minntubes,maxntubes);
  ntubes2->Sumw2();
  TH1F *asymmetryaux = new TH1F("asymmetryaux","asymmetry",nbins,minasymmetry,maxasymmetry);
  TH1F *asymmetry1 = new TH1F("asymmetry1","asymmetry",nbins,minasymmetry,maxasymmetry);
  asymmetry1->Sumw2();
  TH1F *asymmetry2 = new TH1F("asymmetry2","asymmetry",nbins,minasymmetry,maxasymmetry);
  asymmetry2->Sumw2();
  TH1F *tgrad_xaux = new TH1F("tgrad_xaux","tgrad_x",nbins,mintgrad_x,maxtgrad_x);
  TH1F *tgrad_x1 = new TH1F("tgrad_x1","tgrad_x",nbins,mintgrad_x,maxtgrad_x);
  tgrad_x1->Sumw2();
  TH1F *tgrad_x2 = new TH1F("tgrad_x2","tgrad_x",nbins,mintgrad_x,maxtgrad_x);
  tgrad_x2->Sumw2();
  TH1F *cosphiaux = new TH1F("cosphiaux","cosphi",nbins,mincosphi,maxcosphi);
  TH1F *cosphi1 = new TH1F("cosphi1","cosphi",nbins,mincosphi,maxcosphi);
  cosphi1->Sumw2();
  TH1F *cosphi2 = new TH1F("cosphi2","cosphi",nbins,mincosphi,maxcosphi);
  cosphi2->Sumw2();
  TH1F *sinphiaux = new TH1F("sinphiaux","sinphi",nbins,minsinphi,maxsinphi);
  TH1F *sinphi1 = new TH1F("sinphi1","sinphi",nbins,minsinphi,maxsinphi);
  sinphi1->Sumw2();
  TH1F *sinphi2 = new TH1F("sinphi2","sinphi",nbins,minsinphi,maxsinphi);
  sinphi2->Sumw2();

  //create event-wise histograms
  TH1F *MCe0aux = new TH1F("MCe0aux","MCe0",nbins,minMCe0,maxMCe0);
  TH1F *MCe01 = new TH1F("MCe01","MCe0",nbins,minMCe0,maxMCe0);
  MCe01->Sumw2();
  TH1F *MCe02 = new TH1F("MCe02","MCe0",nbins,minMCe0,maxMCe0);
  MCe02->Sumw2();

  TH1F *MCxcoreaux = new TH1F("MCxcoreaux","MCxcore",nbins,minMCxcore,maxMCxcore);
  TH1F *MCxcore1 = new TH1F("MCxcore1","MCxcore",nbins,minMCxcore,maxMCxcore);
  MCxcore1->Sumw2();
  TH1F *MCxcore2 = new TH1F("MCxcore2","MCxcore",nbins,minMCxcore,maxMCxcore);
  MCxcore2->Sumw2();

  TH1F *MCycoreaux = new TH1F("MCycoreaux","MCycore",nbins,minMCycore,maxMCycore);
  TH1F *MCycore1 = new TH1F("MCycore1","MCycore",nbins,minMCycore,maxMCycore);
  MCycore1->Sumw2();
  TH1F *MCycore2 = new TH1F("MCycore2","MCycore",nbins,minMCycore,maxMCycore);
  MCycore2->Sumw2();

  TH1F *MCxoffaux = new TH1F("MCxoffaux","MCxoff",nbins,minMCxoff,maxMCxoff);
  TH1F *MCxoff1 = new TH1F("MCxoff1","MCxoff",nbins,minMCxoff,maxMCxoff);
  MCxoff1->Sumw2();
  TH1F *MCxoff2 = new TH1F("MCxoff2","MCxoff",nbins,minMCxoff,maxMCxoff);
  MCxoff2->Sumw2();

  TH1F *MCyoffaux = new TH1F("MCyoffaux","MCyoff",nbins,minMCyoff,maxMCyoff);
  TH1F *MCyoff1 = new TH1F("MCyoff1","MCyoff",nbins,minMCyoff,maxMCyoff);
  MCyoff1->Sumw2();
  TH1F *MCyoff2 = new TH1F("MCyoff2","MCyoff",nbins,minMCyoff,maxMCyoff);
  MCyoff2->Sumw2();

  //  int xdiv = 1;
  //  int ydiv = 1;

  //   TCanvas *cLST = new TCanvas("LSTs");
  //cLST->Divide(xdiv,ydiv);
  // TCanvas *cMST = new TCanvas("MSTs");
  //cMST->Divide(xdiv,ydiv);

  //for (int i = 1; i < ntel1+1; i++){
  
  for (int i = ntel_0; i < ntel_f+1; i++){
   
    TString st; 
    st.Form("Tel_%d/tpars",i);
    cout << i <<" "<< st << endl;

    TChain ch1(st);
    ch1.Add(mypath1);
    TChain ch2(st);
    ch2.Add(mypath2);

    ch1.Draw("log10(size)>>sizeaux","size>0");
    size1->Add(sizeaux);
    cout << sizeaux->GetEntries() <<endl;
    sizeaux->Reset();
    ch1.Draw("length>>lengthaux","size>0");
    length1->Add(lengthaux);
    lengthaux->Reset();
    ch1.Draw("width>>widthaux","size>0");
    width1->Add(widthaux);
    widthaux->Reset();
    ch1.Draw("meanPedvar_Image>>pedvaraux","size>0");
    pedvar1->Add(pedvaraux);
    pedvaraux->Reset();
    ch1.Draw("cen_x>>cen_xaux","size>0");
    cen_x1->Add(cen_xaux);
    cen_xaux->Reset();
    ch1.Draw("cen_y>>cen_yaux","size>0");
    cen_y1->Add(cen_yaux);
    cen_yaux->Reset();
    ch1.Draw("dist>>distaux","size>0");
    dist1->Add(distaux);
    distaux->Reset();
    ch1.Draw("ntubes>>ntubesaux","size>0");
    ntubes1->Add(ntubesaux);
    ntubesaux->Reset();
    ch1.Draw("asymmetry>>asymmetryaux","size>0");
    asymmetry1->Add(asymmetryaux);
    asymmetryaux->Reset();
    ch1.Draw("tgrad_x>>tgrad_xaux","size>0");
    tgrad_x1->Add(tgrad_xaux);
    tgrad_xaux->Reset();
    ch1.Draw("cosphi>>cosphiaux","size>0");
    cosphi1->Add(cosphiaux);
    cosphiaux->Reset();
    ch1.Draw("sinphi>>sinphiaux","size>0");
    sinphi1->Add(sinphiaux);
    sinphiaux->Reset();
    ch2.Draw("log10(size)>>sizeaux","size>0");
    size2->Add(sizeaux);
    cout << sizeaux->GetEntries() <<endl;
    sizeaux->Reset();
    ch2.Draw("length>>lengthaux","size>0");
    length2->Add(lengthaux);
    lengthaux->Reset();
    ch2.Draw("width>>widthaux","size>0");
    width2->Add(widthaux);
    widthaux->Reset();
    ch2.Draw("meanPedvar_Image>>pedvaraux","size>0");
    pedvar2->Add(pedvaraux);
    pedvaraux->Reset();
    ch2.Draw("cen_x>>cen_xaux","size>0");
    cen_x2->Add(cen_xaux);
    cen_xaux->Reset();
    ch2.Draw("cen_y>>cen_yaux","size>0");
    cen_y2->Add(cen_yaux);
    cen_yaux->Reset();
    ch2.Draw("dist>>distaux","size>0");
    dist2->Add(distaux);
    distaux->Reset();
    ch2.Draw("ntubes>>ntubesaux","size>0");
    ntubes2->Add(ntubesaux);
    ntubesaux->Reset();
    ch2.Draw("asymmetry>>asymmetryaux","size>0");
    asymmetry2->Add(asymmetryaux);
    asymmetryaux->Reset();
    ch2.Draw("tgrad_x>>tgrad_xaux","size>0");
    tgrad_x2->Add(tgrad_xaux);
    tgrad_xaux->Reset();
    ch2.Draw("cosphi>>cosphiaux","size>0");
    cosphi2->Add(cosphiaux);
    cosphiaux->Reset();
    ch2.Draw("sinphi>>sinphiaux","size>0");
    sinphi2->Add(sinphiaux);
    sinphiaux->Reset();
  }

  for (int i = ntel_0; i < ntel_f+1; i++){
    
    TString st = "showerpars";
    cout << i <<" "<< st << endl;

    TChain ch1(st);
    ch1.Add(mypath1);
    TChain ch2(st);
    ch2.Add(mypath2);    
    
    ch1.Draw("MCe0>>MCe0aux");
    MCe01->Add(MCe0aux);
    MCe0aux->Reset();

    ch1.Draw("MCxcore>>MCxcoreaux");
    MCxcore1->Add(MCxcoreaux);
    MCxcoreaux->Reset();
   
    ch1.Draw("MCycore>>MCycoreaux");
    MCycore1->Add(MCycoreaux);
    MCycoreaux->Reset();

    ch1.Draw("MCxoff>>MCxoffaux");
    MCxoff1->Add(MCxoffaux);
    MCxoffaux->Reset();

    ch1.Draw("MCyoff>>MCyoffaux");
    MCyoff1->Add(MCyoffaux);
    MCyoffaux->Reset();

    ch2.Draw("MCe0>>MCe0aux");
    MCe02->Add(MCe0aux);
    MCe0aux->Reset();

    ch2.Draw("MCxcore>>MCxcoreaux");
    MCxcore2->Add(MCxcoreaux);
    MCxcoreaux->Reset();
   
    ch2.Draw("MCycore>>MCycoreaux");
    MCycore2->Add(MCycoreaux);
    MCycoreaux->Reset();

    ch2.Draw("MCxoff>>MCxoffaux");
    MCxoff2->Add(MCxoffaux);
    MCxoffaux->Reset();

    ch2.Draw("MCyoff>>MCyoffaux");
    MCyoff2->Add(MCyoffaux);
    MCyoffaux->Reset();

  }




  if (!plotHistos(size1,size2,false,true,true,tellabel,label1,label2,tag,true)) return -1;
  if (!plotHistos(size1,size2,true,true,true,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(length1,length2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(length1,length2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(width1,width2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(width1,width2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(pedvar1,pedvar2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(pedvar1,pedvar2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(cen_x1,cen_x2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(cen_x1,cen_x2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(cen_y1,cen_y2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(cen_y1,cen_y2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(dist1,dist2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(dist1,dist2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(ntubes1,ntubes2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(ntubes1,ntubes2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(asymmetry1,asymmetry2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(asymmetry1,asymmetry2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(tgrad_x1,tgrad_x2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(tgrad_x1,tgrad_x2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(cosphi1,cosphi2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(cosphi1,cosphi2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(sinphi1,sinphi2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(sinphi1,sinphi2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  
  if (!plotHistos(length1,length2,true,false,false,"dummy","dummy","dummy","dummy",false)) return -1;
  
  if (!plotHistos(MCe01,MCe02,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(MCe01,MCe02,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(MCxcore1,MCxcore2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(MCxcore1,MCxcore2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(MCycore1,MCycore2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(MCycore1,MCycore2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(MCxoff1,MCxoff2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(MCxoff1,MCxoff2,true,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(MCyoff1,MCyoff2,false,true,false,tellabel,label1,label2,tag,false)) return -1;
  if (!plotHistos(MCyoff1,MCyoff2,true,true,false,tellabel,label1,label2,tag,false)) return -1;



  /*  size1->Reset();  
  size2->Reset();  
  length1->Reset();
  length2->Reset();


  
  for (int j = ntel1+1; j < (ntel1+ntel2)+1; j++){
    st.Form("Tel_%d/tpars",j);
    cout << j <<" "<< st << endl;
    
    TChain ch1(st);
    ch1.Add(mypath1);
    ch1.Draw("size>>sizeaux","size>0");
    size1->Add(sizeaux);
    ch1.Draw("length>>lengthaux","size>0");
    length1->Add(lengthaux);
    cout << sizeaux->GetEntries() <<endl;

    TChain ch2(st);
    ch2.Add(mypath2);
    ch2.Draw("size>>sizeaux","size>0");
    size2->Add(sizeaux);
    ch2.Draw("length>>lengthaux","size>0");
    length2->Add(lengthaux);
    cout << sizeaux->GetEntries() <<endl;
    
  }
  */
  /*
  if (!plotHistos(size1,size2,false,true,true,tel2label)) return -1;
  if (!plotHistos(length1,length2,false,true,false,tel2label)) return -1;
  if (!plotHistos(length1,length2,true,true,false,tel2label)) return -1;
  */
  return 0;
}
bool plotHistos(TH1F *h1, TH1F *h2, bool bnorm, bool bsave, bool blog, TString label, TString label1, TString label2, TString tag,bool debug){
  TCanvas *c1 = new TCanvas();
  c1->cd();
  TPad *pad1 = new TPad("pad1", "pad1", 0, 0.4, 1, 1.0);
  pad1->SetBottomMargin(0.02); // Upper and lower plot are joined                                
  pad1->SetGridx();         // Vertical grid                                                  
  if (blog){
    //pad1->SetLogx();
    pad1->SetLogy();
  }
  pad1->Draw();             // Draw the upper pad: pad1                                       
  pad1->cd();               // pad1 becomes the current pad                                   
  h1->SetStats(0);          // No statistics on upper plot
  int inte1 = h1->Integral();
  if (bnorm) h1->Scale(1/float(inte1));    
  h1->Draw();
  // if (bnorm) h2->Scale(1/h2->Integral(), "width");                                
  int inte2 = h2->Integral();
  if (bnorm) h2->Scale(1/float(inte2));                                
  h2->Draw("same");
  h1->GetXaxis()->SetLabelSize(0.);

  TLegend* leg = new TLegend(0.7,0.7,0.85,0.87);
  leg->AddEntry((TObject*)0,Form("%s",label.Data()),""); // option "C" allows to center the header
  leg->AddEntry((TObject*)0,Form("%s",tag.Data()),""); // option "C" allows to center the header
  leg->AddEntry(h1,Form("%s",label1.Data()),"l");
  leg->AddEntry(h2,Form("%s",label2.Data()),"l");
  leg->SetBorderSize(0);
  leg->SetTextFont(43);
  leg->Draw();
  if (bnorm){
    TLegend* auxleg = new TLegend(0.7,0.6,0.85,0.7);
    auxleg->SetBorderSize(0);
    auxleg->SetTextFont(43);
    auxleg->AddEntry((TObject*)0,"(Normalized)", "");
    auxleg->Draw();
  }
      /*TGaxis *axis = new TGaxis( -5, 20, -5, 220, 20,220,510,"");
  axis->SetLabelFont(43); // Absolute font size in pixel (precision 3)                        
  axis->SetLabelSize(15);
  axis->Draw();*/
  c1->cd();

  TPad *pad2 = new TPad("pad2", "pad2", 0, 0, 1, 0.4);
  pad2->SetTopMargin(0);
  pad2->SetBottomMargin(0.1);
  pad2->SetGridx(); // vertical grid
  //if (blog) pad2->SetLogx();                                                          
  pad2->Draw();
  pad2->cd();       // pad2 becomes the current pad   
  TH1F *ratio = (TH1F*)h1->Clone(h1->GetTitle());
  ratio->SetLineColor(kBlack);
  //  ratio->SetMinimum(0.8);  // Define Y ..  
  //  ratio->SetMaximum(1.35); // .. range                       
  ratio->Sumw2();
  ratio->SetStats(0);      // No statistics on lower plot      
  ratio->Divide(h2);
  ratio->SetMarkerStyle(21);
  ratio->Draw("ep");       // Draw the ratio plot     
  TLine *refline = new TLine(ratio->GetXaxis()->GetXmin(),1,ratio->GetXaxis()->GetXmax(),1);
  refline->SetLineColor(kRed);
  refline->Draw();
  TLegend *leg2 = new TLegend(0.25,0.75,0.85,0.95);
  leg2->AddEntry((TObject*)0, Form("%s / %s: %d imgs / %d imgs",label1.Data(),label2.Data(),inte1,inte2), "");
  //leg2->AddEntry((TObject*)0, Form("GTech: %d evts",inte2), "");
  float vratio = float(inte1)/float(inte2);
  float vratio_err = sqrt(pow(sqrt(inte1)/inte2,2)+pow(inte1*sqrt(inte2)/pow(inte2,2),2));
  leg2->AddEntry((TObject*)0, Form("Ratio: %1.3f +/- %1.3f",vratio,vratio_err), "");
  leg2->SetBorderSize(0);
  leg2->SetTextFont(43);
  leg2->Draw();

  // h1 settings                                                                              
  h1->SetLineColor(kBlue+1);
  h1->SetLineWidth(2);

  // Y axis h1 plot settings                                                                  
  h1->GetYaxis()->SetTitleSize(20);
  h1->GetYaxis()->SetTitleFont(43);
  h1->GetYaxis()->SetTitleOffset(1.55);
  
  h1->GetYaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)   
  h1->GetYaxis()->SetLabelSize(15);
  if (!blog) h1->GetYaxis()->SetRangeUser(TMath::Min(h1->GetMinimum(),h2->GetMinimum()),1.1*TMath::Max(h1->GetMaximum(),h2->GetMaximum()));

  // h2 settings                                                                              
  h2->SetLineColor(kRed);
  h2->SetLineWidth(2);

  // Ratio plot (ratio) settings                                                                 
  ratio->SetTitle(""); // Remove the ratio title                                                 

  // Y axis ratio plot settings                                                               
  ratio->GetYaxis()->SetTitle(Form("%s/%s",label1.Data(),label2.Data()));
  ratio->GetYaxis()->SetRangeUser(0.5,2);
  // ratio->GetYaxis()->SetNdivisions(505);
  ratio->GetYaxis()->SetTitleSize(15);
  ratio->GetYaxis()->SetTitleFont(43);
  ratio->GetYaxis()->SetTitleOffset(1.55);
  ratio->GetYaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
  ratio->GetYaxis()->SetLabelSize(15);

  // X axis ratio plot settings                                                               
  ratio->GetXaxis()->SetTitleSize(20);
  ratio->GetXaxis()->SetTitleFont(43);
  ratio->GetXaxis()->SetTitleOffset(4.);
  ratio->GetXaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)  
  ratio->GetXaxis()->SetLabelSize(15);

  if (bsave){
    if (bnorm){
      c1->SaveAs(Form("%s_%s_%s_%s_vs_%s_norm.eps",label.Data(),tag.Data(),h1->GetTitle(),label1.Data(),label2.Data()));
      c1->SaveAs(Form("%s_%s_%s_%s_vs_%s_norm.png",label.Data(),tag.Data(),h1->GetTitle(),label1.Data(),label2.Data()));
    }    
    else {
      c1->SaveAs(Form("%s_%s_%s_%s_vs_%s.eps",label.Data(),tag.Data(),h1->GetTitle(),label1.Data(),label2.Data()));
      c1->SaveAs(Form("%s_%s_%s_%s_vs_%s.png",label.Data(),tag.Data(),h1->GetTitle(),label1.Data(),label2.Data()));
    }
  }
  
  int aux1 = 0;
  int aux2 = 0;
  if (debug){
    cout <<"index h1 h2 ratio1 ratio2 lowEdge highEdge"<<endl;
    for (int k=1; k<h1->GetNbinsX(); k++){
      if (h2->GetBinContent(k)>0){
      cout << k <<" "<< h1->GetBinContent(k)<<" "<<h2->GetBinContent(k)<<" "<<h1->GetBinContent(k)/h2->GetBinContent(k)<<" "<<ratio->GetBinContent(k)<<" "<<h1->GetBinLowEdge(k)<<" "<<h1->GetBinLowEdge(k)+h1->GetBinWidth(k) <<endl;
      }
      else{
cout << k <<" "<< h1->GetBinContent(k)<<" "<<h2->GetBinContent(k)<<" "<<"NaN"<<" "<<ratio->GetBinContent(k)<<" "<<h1->GetBinLowEdge(k)<<" "<<h1->GetBinLowEdge(k)+h1->GetBinWidth(k) <<endl;
      }


      aux1 = aux1 + h1->GetBinContent(k);
      aux2 = aux2 + h2->GetBinContent(k);
    }
    cout <<"aux1 inte1 aux2 inte2" <<endl;
    cout << aux1 << " " << inte1 << " " << aux2 << " " << inte2 << " " <<endl;
  }
  
  return true;
}

int main(int argc, char * argv[])
{
    //TString mypath1 = "/data/nieto/deeplearning/evn/gamma/20deg/0deg/root";
    TString mypath1 = "/data/deeplearning/data/root-evndisp/gamma-diffuse/*.root";
    TString label1 = "gamma-diffuse";


    //TString mypath2 = "/data/nieto/deeplearning/evn/gamma/20deg/180deg/*root";
    TString mypath2 = "/data/deeplearning/data/root-evndisp/proton/*.root";
    TString label2 = "proton";
    
    TString tag = "particle";

    int teltoana = 1;

    return plotImgPars(mypath1,label1,mypath2,label2,tag,teltoana);
}
