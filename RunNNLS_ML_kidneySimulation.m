%give signal input, will fit and output the peaks
% can also give pt number and ROI type

%written to work on simulated three compartment kidney ellipsoids


% Example code to run
%{
twopeak_example = [1, 0.836735, 0.77551, 0.744898, 0.663265, 0.520408, 0.469388, 0.153061, 0.0714286]';
bvals = [0, 10, 30, 50, 80, 120, 200, 400, 800];
lambda = 8
[OutputDiffusionSpectrum, rsq, Resid, y_recon, resultsPeaks] = RunNNLS_ML_kidneySimulation(twopeak_example, bvals, lambda);
Export_Cell = {'example 2peak simpleCVNNLS lambda 8', resultsPeaks',rsq, OutputDiffusionSpectrum'};
ExcelFileName=['/Users/miraliu/Desktop/PostDocCode/Multiexp_Simulations_python/RA_Spectra.xlsx'];
writecell(Export_Cell,ExcelFileName,'WriteMode','append','Sheet','SpectralPlots')

% set lambda by hand... 

threepeak_example = [1, 0.935484, 0.822581, 0.798387, 0.73387, 0.580645, 0.419355, 0.274194, 0.129032]';
bvals = [0, 10, 30, 50, 80, 120, 200, 400, 800];
lambda = 8;
[OutputDiffusionSpectrum, rsq, Resid, y_recon, resultsPeaks] = RunNNLS_ML_kidneySimulation(threepeak_example, bvals, lambda);
Export_Cell = {'example 3peak simpleCVNNLS lambda 8', resultsPeaks',rsq, OutputDiffusionSpectrum'};
ExcelFileName=['/Users/miraliu/Desktop/PostDocCode/Multiexp_Simulations_python/RA_Spectra.xlsx'];
writecell(Export_Cell,ExcelFileName,'WriteMode','append','Sheet','SpectralPlots')
%}

% running on anomalous data test
%{
%11/21/2024
anomalous_example = [0.97756695, 0.85895991, 0.72244476, 0.63996803, 0.55878782,0.48921553, 0.39920355, 0.26036917, 0.1210697 ]';
bvals = [0, 10, 30, 50, 80, 120, 200, 400, 800];
lambda = 8;
[OutputDiffusionSpectrum, rsq, Resid, y_recon, resultsPeaks] = RunNNLS_ML_kidneySimulation(anomalous_example, bvals, lambda);
Export_Cell = {'anomalous 3peak simpleCVNNLS lambda 8', resultsPeaks',rsq, OutputDiffusionSpectrum'};
ExcelFileName=['/Users/miraliu/Desktop/PostDocCode/Multiexp_Simulations_python/RA_Spectra.xlsx'];
writecell(Export_Cell,ExcelFileName,'WriteMode','append','Sheet','SpectralPlots')


% running on ideal 
%11/21/24
ideal_example = [1.        , 0.87867133, 0.73877401, 0.65509785, 0.57202138,0.49919888, 0.40768673, 0.26969802, 0.12113794]';
bvals = [0, 10, 30, 50, 80, 120, 200, 400, 800];
lambda = 8;
[OutputDiffusionSpectrum, rsq, Resid, y_recon, resultsPeaks] = RunNNLS_ML_kidneySimulation(ideal_example, bvals, lambda);
Export_Cell = {'ideal 3peak simpleCVNNLS lambda 8', resultsPeaks',rsq, OutputDiffusionSpectrum'};
ExcelFileName=['/Users/miraliu/Desktop/PostDocCode/Multiexp_Simulations_python/RA_Spectra.xlsx'];
writecell(Export_Cell,ExcelFileName,'WriteMode','append','Sheet','SpectralPlots')
%}

function [OutputDiffusionSpectrum, rsq, Resid, y_recon, SortedresultsPeaks] = RunNNLS_ML_kidneySimulation(varargin)

    addpath /Users/miraliu/Desktop/PostDocCode/Applied_NNLS_renal_DWI/rNNLS/nwayToolbox
    addpath /Users/miraliu/Desktop/PostDocCode/Applied_NNLS_renal_DWI/rNNLS
    addpath /Users/miraliu/Desktop/PostDocCode/DSI_IVIM_Maps/Kidney_DSI
%    disp(PatientNum)

    %list_of_b_values = zeros(length(bvalues),max(bvalues));
    %list_of_b_values(h,1:length(b_values)) = b_values; %make matrix of b-values
    %b_values = [0,10,30,50,80,120,200,400,800];

    SignalInput = varargin{1};
    b_values = varargin{2};
    lambda = varargin{3};

    %% Generate NNLS space of values, not entirely sure about this part, check with TG?
    ADCBasisSteps = 300; %(??)
    ADCBasis = logspace( log10(5), log10(2200), ADCBasisSteps);
    A = exp( -kron(b_values',1./ADCBasis));

    
    %% create empty arrays to fill
    amplitudes = zeros(ADCBasisSteps,1);
    resnorm = zeros(1);
    resid = zeros(length(b_values),1);
    y_recon = zeros(max(b_values),1);
    resultsPeaks = zeros(6,1); %6 was 9 before? unsure why

    
   


    %% try to git them with NNLS
    %[TempAmplitudes, TempResnorm, TempResid ] = CVNNLS(A, SignalInput);
    
    %% with forced regularization of curve
    [TempAmplitudes, TempResnorm, TempResid ] = simpleCVNNLS_curveregularized(A, SignalInput, lambda);
    

    amplitudes(:) = TempAmplitudes';
    resnorm(:) = TempResnorm';
    resid(1:length(TempResid)) = TempResid';
    y_recon(1:size(A,1)) = A * TempAmplitudes;

    % to match r^2 from bi-exp, check w/ octavia about meaning of this 
    SSResid = sum(resid.^2);
    SStotal = (length(b_values)-1) * var(SignalInput);
    rsq = 1 - SSResid/SStotal; 

    


    %% output renaming, just to stay consistent with the TG&JP code
    OutputDiffusionSpectrum = amplitudes;
    %plot(OutputDiffusionSpectrum);
    %pause(1)
    Chi = resnorm;
    Resid = resid;

    %attempt with TG version? prior to TG meeting Sept 14th. 
    % assumed ADC thresh from 2_Simulation...
    ADCThresh = 1./sqrt([0.180*0.0058 0.0058*0.0015]);
    %[GeoMeanRegionADC_1,GeoMeanRegionADC_2,GeoMeanRegionADC_3,RegionFraction1,RegionFraction2,RegionFraction3 ] = NNLS_resultTG(OutputDiffusionSpectrum, ADCBasis, ADCThresh);

    [GeoMeanRegionADC_1,GeoMeanRegionADC_2,GeoMeanRegionADC_3,GeoMeanRegionADC_4,RegionFraction1,RegionFraction2,RegionFraction3,RegionFraction4 ] = NNLS_result_mod_ML_fourpeaks(OutputDiffusionSpectrum, ADCBasis);
    resultsPeaks(1) = RegionFraction1; %(frac_fast - RegionFraction1)./frac_fast.*100;
    resultsPeaks(2) = RegionFraction2; %(frac_med - RegionFraction2)./frac_med.*100;
    resultsPeaks(3) = RegionFraction3; %(frac_slow - )./frac_slow.*100;
    resultsPeaks(4) = RegionFraction4; %(frac_fibro - )./frac_slow.*100;
    resultsPeaks(5) = GeoMeanRegionADC_1; %(diff_fast - GeoMeanRegionADC_1./1000)./diff_fast.*100;
    resultsPeaks(6) = GeoMeanRegionADC_2; %(diff_med - GeoMeanRegionADC_2./1000)./diff_med.*100;
    resultsPeaks(7) = GeoMeanRegionADC_3; %(diff_slow - GeoMeanRegionADC_3./1000)./diff_slow.*100;
    resultsPeaks(8) = GeoMeanRegionADC_4; %(diff_fibro - GeoMeanRegionADC_3./1000)./diff_slow.*100;



    SortedresultsPeaks = ReSort_fourpeaks(resultsPeaks);


end


