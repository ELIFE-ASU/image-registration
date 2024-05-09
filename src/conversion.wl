(* ::Package:: *)

BeginPackage["Conversion`"]


FlatFieldCorrect::usage = "Subtract the psuedo-flat field from the frames of a video"


VideoCrop::usage = "Crop the frames of a video with padding"


OutDir::usage = "Determine the output directory for an analysis"


Begin["`Private`"]


FlatField[images_] := GaussianFilter[#, ImageDimensions[#]]& @ Mean[images]


FlatFieldCorrect[phase_] := With[{ff = FlatField[phase]},
	ImageDifference[#, ff]& /@ phase]


VideoCrop[frames_, pad_] := With[{border = BorderDimensions@First@frames, aspect = ImageAspectRatio@First@frames},
    ImagePad[#, -border + pad * {{aspect, 1}, {aspect, 1}}]& /@ frames
]


OutDir[dirname_,filename_, subdirs___]:=FileNameJoin[{dirname, FileBaseName[filename], subdirs}]
OutDir[dirname_,filename_, phase_Integer]:= OutDir[dirname, filename, "phase_" <> ToString[phase]]


End[]


EndPackage[]
