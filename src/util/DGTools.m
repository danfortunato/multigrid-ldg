(* ::Package:: *)

(* ::Title:: *)
(*DGTools*)


(* ::Subtitle:: *)
(*Dan Fortunato*)


(* ::Section::Closed:: *)
(*Begin*)


BeginPackage["DGTools`"];

DGFunction::usage = "";
DGLoad::usage = "";
DGRun::usage = "";
DGConvergence::usage = "";

DGPlot::usage = "";
DGConvergencePlot::usage = "";

Begin["`Private`"];


$baseDir = FileNameJoin[{$HomeDirectory, "Dropbox", "Documents", "Saye", "multigrid-ldg"}];
$binDir  = FileNameJoin[{$baseDir, "bin"}];


(* ::Section::Closed:: *)
(*DGFunction*)


(* ::Subsection::Closed:: *)
(*Formatting*)


DGFunction /: MakeBoxes[DGFunction[{P_, N_, ne_}, data_, internal___], StandardForm] := With[
	{icon = ListDensityPlot[data, Frame->False, PlotRangePadding->None, PlotRangeClipping->None, ImageSize->{Automatic,Dynamic[3.5*(CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification])]}]},
	BoxForm`ArrangeSummaryBox[DGFunction, data, icon, {
		BoxForm`MakeSummaryItem[{"Polynomial order: ", P-1}, StandardForm],
		BoxForm`MakeSummaryItem[{"Number of elements: ", ne}, StandardForm]
	},
	{BoxForm`MakeSummaryItem[{"Dimension: ", N}, StandardForm]},
	StandardForm
]]


(* ::Subsection::Closed:: *)
(*Import*)


(* ::Subsubsection::Closed:: *)
(*Registration*)


ImportExport`RegisterImport[
	"DGFunction",
	DGTools`ImportDGFunction,
	"DefaultElement" -> "Data"
]


(* ::Subsubsection::Closed:: *)
(*Implementation*)


ImportDGFunction[file_String, opts___] := Module[{stream, data, header, elems},
	stream = OpenRead[file];
	data = ReadString[stream];
	data = StringReplace[data, {"e+" :> "*^", "e-" :> "*^-"}];
	{header, elems} = StringSplit[data, "\n", 2];
	header = StringSplit[header] // ToExpression;
	elems = StringSplit /@ StringSplit[StringSplit[elems, "\n\n"], "\n"] // ToExpression;
	Close[stream];
	{
		"Header" -> header,
		"Data" -> DGFunction[header, elems]
	}
]


(* ::Subsection::Closed:: *)
(*Info*)


DGFunction[{P_, N_, ne_}, data_, internal___]["P"] := P;
DGFunction[{P_, N_, ne_}, data_, internal___]["N"] := N;
DGFunction[{P_, N_, ne_}, data_, internal___]["NumberOfElements"] := ne;
DGFunction[{P_, N_, ne_}, data_, internal___]["Data"] := data;
DGFunction[{P_, N_, ne_}, data_, internal___]["Mesh"] := data[[All, All, ;;N]];
DGFunction[{P_, N_, ne_}, data_, internal___]["Coeffs"] := data[[All, All, -1]];
DGFunction[{P_, N_, ne_}, data_, internal___]["Interpolants"] := internal;


(* ::Subsection::Closed:: *)
(*Evaluation*)


DGFunction[{P_, N_, ne_}, data_] := Module[{interpolants},
	interpolants = Function[elem,
		<|"Interpolant" -> Interpolation[{Most[#], Last[#]}& /@ elem, InterpolationOrder -> P-1],
		  "Domain" -> BoundingRegion[Most /@ elem]|>]
	/@ data;
	DGFunction[{P, N, ne}, data, interpolants]
]


DGFunction[{P_, N_, ne_}, data_, interpolants_][x__?NumericQ] /; (Length[{x}] === N) := Module[{e, nodes, coeffs, \[Phi]},
	nodes = Map[Most, data, {2}];
	e = FirstPosition[RegionMember[#[["Domain"]], {x}]& /@ interpolants, True] // First;
	interpolants[[e,"Interpolant"]][x]
]


(* ::Subsection::Closed:: *)
(*UpValues*)


(* ::Subsubsection::Closed:: *)
(*Dimensions*)


DGFunction /: Dimensions[f:DGFunction[{P_, N_, ne_}, ___]] := {ne, P, N}


(* ::Subsubsection::Closed:: *)
(*Dot*)


DGFunction /: Dot[A__?MatrixQ, f:DGFunction[{P_, N_, ne_}, _, ___]] := Module[{g, nc},
	g = Dot[A, Flatten[f["Coeffs"]]];
	nc = Length[g]/(P^N*ne);
	If[nc === 1,
		DGFunction[{P, N, ne}, MapThread[Join, {f["Mesh"], Partition[List/@g,P^N]}, 2]],
		Table[DGFunction[{P, N, ne}, MapThread[Join, {f["Mesh"], Partition[List/@g,P^N][[i;;;;nc]]}, 2]], {i, nc}]
	]
]


(* ::Subsubsection::Closed:: *)
(*Norm*)


DGFunction /: Norm[f_DGFunction] := Norm[f, \[Infinity]];


DGFunction /: Norm[f_DGFunction, \[Infinity]] := Max[Abs[f["Coeffs"]]];


(* ::Subsubsection::Closed:: *)
(*Plus*)


DGFunction /: f_DGFunction + a_?NumericQ := Module[{data},
	data = f["Data"];
	data[[All,All,-1]] += a;
	DGFunction[{f["P"], f["N"], f["NumberOfElements"]}, data]
]


DGFunction /: f_DGFunction + g_DGFunction /; f["Mesh"] === g["Mesh"] := Module[{data},
	data = f["Data"];
	data[[All,All,-1]] += g["Coeffs"];
	DGFunction[{f["P"], f["N"], f["NumberOfElements"]}, data]
]


(* ::Subsubsection::Closed:: *)
(*Times*)


DGFunction /: a_?NumericQ * f_DGFunction := Module[{data},
	data = f["Data"];
	data[[All,All,-1]] *= a;
	DGFunction[{f["P"], f["N"], f["NumberOfElements"]}, data]
]


DGFunction /: f_DGFunction * g_DGFunction /; f["Mesh"] === g["Mesh"] := Module[{data},
	data = f["Data"];
	data[[All,All,-1]] *= g["Coeffs"];
	DGFunction[{f["P"], f["N"], f["NumberOfElements"]}, data]
]


(* ::Subsubsection::Closed:: *)
(*Power*)


DGFunction /: f_DGFunction^a_?NumericQ := Module[{data},
	data = f["Data"];
	data[[All,All,-1]] = data[[All,All,-1]]^a;
	DGFunction[{f["P"], f["N"], f["NumberOfElements"]}, data]
]


(* ::Subsubsection::Closed:: *)
(*LinearSolve*)


DGFunction /: LinearSolve[A_?MatrixQ, f:DGFunction[{P_, N_, ne_}, _, ___], opts:OptionsPattern[LinearSolve]] := Module[{u, data},
	u = LinearSolve[A, Flatten[f["Coeffs"]], opts];
	data = MapThread[Join, {f["Mesh"], Partition[List/@u, P^N]}, 2];
	DGFunction[{f["P"], f["N"], f["NumberOfElements"]}, data]
]


(* ::Subsubsection::Closed:: *)
(*LeastSquares*)


DGFunction /: LeastSquares[A_?MatrixQ, f:DGFunction[{P_, N_, ne_}, _, ___], opts:OptionsPattern[LeastSquares]] := Module[{u, data},
	u = LeastSquares[A, Flatten[f["Coeffs"]], opts];
	data = MapThread[Join, {f["Mesh"], Partition[List/@u, P^N]}, 2];
	DGFunction[{f["P"], f["N"], f["NumberOfElements"]}, data]
]


(* ::Section::Closed:: *)
(*DGLoad*)


DGLoad[] := Module[{M, MM, G, T, u, f, Jg, Jh, aD},
	M  = Import[FileNameJoin[{$baseDir, "M.mtx"}]];
	MM = Import[FileNameJoin[{$baseDir, "MM.mtx"}]];
	G  = Import[FileNameJoin[{$baseDir, "G.mtx"}]];
	T  = Import[FileNameJoin[{$baseDir, "T.mtx"}]];
	u  = Import[FileNameJoin[{$baseDir, "u.fun"}], "DGFunction"];
	f  = Import[FileNameJoin[{$baseDir, "f.fun"}], "DGFunction"];
	Jg = Import[FileNameJoin[{$baseDir, "Jg.mtx"}]];
	Jh = Import[FileNameJoin[{$baseDir, "Jh.mtx"}]];
	aD = Import[FileNameJoin[{$baseDir, "aD.mtx"}]];
	{M, MM, G, T, u, f, Jg, Jh, aD}
]


(* ::Section::Closed:: *)
(*DGRun*)


DGRun[h_:Nothing, bctype_:Nothing, \[Tau]0_:Nothing, \[Tau]D_:Nothing] := Run[
	FileNameJoin[{$binDir, "dg_tests"}],
	Sequence@@{N[h], bctype, N[\[Tau]0], N[\[Tau]D]}
]


(* ::Section::Closed:: *)
(*DGConvergence*)


DGConvergence[hs:{__?NumericQ}, bctype_:Nothing, \[Tau]0_:Nothing, \[Tau]D_:Nothing] := Module[
	{u, uu, f, M, MM, G, T, solve},
	solve = If[bctype===2, LeastSquares, LinearSolve];
	Reap[Do[
		DGRun[h, bctype, \[Tau]0, \[Tau]D];
		{M, MM, G, T, u, f} = DGLoad[];
		uu = solve[G\[Transpose].MM.G+T, M.f];
		Sow[Norm[uu - u] / Norm[u]],
		{h, N[hs]}
	]][[2, 1]]
]


(* ::Section::Closed:: *)
(*DGPlot*)


(*DGPlot[f_DGFunction, opts:OptionsPattern[Plot3D]] /; (f["N"] === 2) := Module[{ints, x, y},
	ints = f["Interpolants"];
	Show[##, PlotRange -> All]& @@ Table[
		Plot3D[ints[[i,"Interpolant"]][x, y], {x,y}\[Element]ints[[i,"Domain"]], opts],
		{i, 1, f["NumberOfElements"]}
	]
]*)


DGPlot[f_DGFunction, opts:OptionsPattern[ListPlot3D]] := Module[{int},
	ListPlot3D[f["Data"], opts]
]


(* ::Section::Closed:: *)
(*DGConvergencePlot*)


DGConvergencePlot[hs:{__?NumericQ}] := DGConvergencePlot[hs, DGConvergence[hs]]


DGConvergencePlot[hs:{__?NumericQ}, err:{__?NumericQ}] := Module[
	{data, fit, x, slope, domain, range, xticks, yticks, format, relabel},

	data = Transpose[{hs, err}];
	fit = Fit[Log[data[[-3;;]]],{1,x},x];
	slope = fit // Last // First;
	fit = fit /. x->Log[x] // Exp;

	xticks = Transpose[{hs, Superscript[2,#]& /@ Round@Log2[hs]}];
	yticks = Range@@{Floor[#1],Ceiling[#2]}& @@ MinMax@Log10[err] // Reverse;
	yticks = {10^#, Superscript[10,#]}& /@ yticks;

	domain = MinMax[hs] + {-1,1}*MinMax[hs]/2;
	range = MinMax[yticks[[All,1]]] + {-1,1}*MinMax[yticks[[All,1]]]/2;

	Show[
		LogLogPlot[fit, {x, First[domain], Last[domain]},
			PlotRange -> {domain, range},
			PlotLegends -> {NumberForm[slope,3]},
			Frame -> True,
			PlotRangePadding -> False,
			FrameTicks -> {{yticks, None}, {xticks, None}}
		],
		ListPlot[Log[data],
			PlotMarkers -> {Automatic, 12},
			PlotRangePadding -> False
		]
	]
]


DGConvergencePlot[hs:{__?NumericQ}, err:{{__?NumericQ}..}] := Module[
	{data, fit, x, slope, domain, range, xticks, yticks},

	data = Transpose[{Take[hs, Length[#]], #}]& /@ err;
	fit = Fit[Log[#[[-3;;-2]]],{1,x},x]& /@ data;
	slope = First[Last[#]]& /@ fit;
	fit = fit /. x->Log[x] // Exp;

	xticks = Transpose[{hs, Superscript[2,#]& /@ Round@Log2[hs]}];
	yticks = Range@@{Floor[#1],Ceiling[#2]}& @@ MinMax@Log10[Flatten@err] // Reverse;
	yticks = {10^#, Superscript[10,#]}& /@ yticks;

	domain = MinMax[hs] + {-1,1}*MinMax[hs]/2;
	range = MinMax[yticks[[All,1]]] + {-1,1}*MinMax[yticks[[All,1]]]/2;

	Show[
		LogLogPlot[fit, {x, First[domain], Last[domain]},
			PlotRange -> {domain, range},
			PlotLegends -> {ToString[NumberForm[#,{2,1}]]& /@ slope},
			Frame -> True,
			PlotRangePadding -> False,
			FrameTicks -> {{yticks, None}, {xticks, None}}
		],
		ListPlot[Log[data],
			PlotMarkers -> {Automatic, 12},
			PlotRangePadding -> False
		]
	]
]


(* ::Section::Closed:: *)
(*End*)


End[];
EndPackage[];
