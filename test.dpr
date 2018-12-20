program test;

{$IFDEF FPC}
{$mode delphi}{$H+}
{$ELSE}
{$APPTYPE CONSOLE}
{$ENDIF}

uses Classes, SysUtils, pastinn;

type
  TTestData = record
    inp: array of TSingleArray; // 2D floating point array of input
    tg: array of TSingleArray; // 2D floating point array of target
    nips: Integer; // Number of inputs to neural network
    nops: Integer; // Number of outputs to neural network
    rows: Integer; // Number of rows in file (number of sets for neural network)
  end;

// Setup data record
function InitData(nips: Integer; nops: Integer; rows: Integer): TTestData;
begin
  SetLength(Result.inp,rows,nips);
  SetLength(Result.tg,rows,nops);
  Result.nips := nips;
  Result.nops := nops;
  Result.rows := rows;
end;

// Parse one row of inputs and outputs
procedure parse(var data: TTestData; line: String; row: Integer);
var
  col, cols: Integer;
  val: Single;
  lparts: TArray;
begin
  cols := data.nips + data.nops;
  lparts := explode(' ',line,0);
  for col := 0 to cols-1 do
  begin
    val := StrToFloat(lparts[col]);
    if col < data.nips then data.inp[row,col] := val
    else data.tg[row,col - data.nips] := val;
  end;
end;

// Randomly shuffles the data
procedure shuffle(var data: TTestData);
var
  a,b: Integer;
  ot, it: TSingleArray;
begin
  for a := 0 to data.rows-1 do
  begin
    b := Random(32767) mod data.rows;
    ot := data.tg[a];
    it := data.inp[a];
    // Swap output
    data.tg[a] := data.tg[b];
    data.tg[b] := ot;
    // Swap input
    data.inp[a] := data.inp[b];
    data.inp[b] := it;
  end;
end;

// Parses file from path getting all inputs and outputs for the neural network
function build(nips: Integer; nops: Integer): TTestData;
var
  t: TStrings;
  row: Integer;
begin
  t := TStringList.Create;
  t.LoadFromFile('semeion.data');
  Result := InitData(nips, nops, t.Count);
  for row := 0 to t.Count-1 do
  begin
    parse(Result,t[row],row);
  end;
  t.Free;
end;

procedure ShowVisual(const data: TTestData);
var
  i,j: Integer;
begin
  for i := 0 to 15 do
  begin
    for j := 0 to 15 do
    begin
      write(data.inp[0][i+j]:1:0,' ');
    end;
    writeln;
  end;
  writeln;
  for i := 0 to 15 do
  begin
    for j := 0 to 15 do
    begin
      write(data.inp[1][i+j]:1:0,' ');
    end;
    writeln;
  end;
end;

procedure main;
var
  nips, nops, nhid, iterations, i, j: Integer;
  rate, anneal, error: Single;
  data: TTestData;
  NN: TTinyNN;
  pd: TSingleArray;
begin
  Randomize;
  // Number of inputs
  nips := 256;
  // Number of outputs
  nops := 10;
  { Learning rate is annealed and thus not constant.
    It can be fine tuned along with the number of hidden layers.
    Feel free to modify the anneal rate.
    The number of iterations can be changed for stronger training. }
  rate := 1.00;
  nhid := 28;
  anneal := 0.99;
  iterations := 128;
  // Load the test data into the test data record
  data := build(nips, nops);
  // Create the Tinn
  NN := TTinyNN.Create;
  // Prepare Tinn
(*  NN.Build(nips, nhid, nops);
  // Train that brain!
  for i := 0 to iterations-1 do
  begin
    shuffle(data);
    error := 0.00;
    for j := 0 to data.rows -1 do
    begin
      error := error + NN.Train(data.inp[j],data.tg[j],rate);
    end;
    writeln('error ',(error/data.rows):1:10, ' :: learning rate ',rate:1:10);
    rate := rate * anneal;
  end;
  { Save to a file
    This is slightly different to the original Tinn project in that it saves
    the hidden output layer weights aswell }
  NN.SaveToFile('test.tinn');*)

  shuffle(data);
  NN.LoadFromFile('test.tinn');

  { Perform a prediction, ideally a test set would be loaded to make the prediction
    with, but for testing purposes we are just reusing the training set loaded
    earlier. One data set is picked at random - as the data was shuffled earlier
    we can just use the first index of the input and target arrays }
  pd := NN.Predict(data.inp[0]);
  // Dump out the target
  NN.PrintToScreen(data.tg[0], data.nops);
  // And finally the prediction
  NN.PrintToScreen(pd, data.nops);
  { If all is well, the prediction that lines up with the target of 1.000000 has
    a value of near to 1 itself }
  NN.Free;
  ShowVisual(data);
end;

begin
  main;
  {$IFNDEF FPC}
  Readln;
  {$ENDIF}
end.
