program test;

(* This test program learns hand written digits and attempts to predict which digit 
   has been passed to the neural network. 
   
   It requires training data available at
   http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data
   
   For more information about creating your own test data please see
   https://www.matthewhipkin.co.uk/blog/into-the-unknown/
*)

{$IFDEF FPC}
{$mode delphi}{$H+}
{$ELSE}
{$APPTYPE CONSOLE}
{$ENDIF}

uses Classes, SysUtils, pastinn;

{ Procedure to load data from file and process into TTinnData record }
procedure BuildData(fn: String; nips: Integer; nops: Integer; var data: TTinnData);
var
  row, rows, col, cols: Integer;
  val: Single;
  lparts: TArray;
  fi: TStrings;
begin
  fi := TStringList.Create;
  { Load data straight into a TStringList - not necessarily the most efficient way
    when dealing with large datasets though! }
  fi.LoadFromFile(fn);
  // Calculate how many rows and columns we need
  cols := nips + nops;
  rows := fi.Count -1;
  // Setup our input and target 2d arrays
  SetLength(data.inp,rows+1,nips);
  SetLength(data.tg,rows+1,nops);
  // Loop each row of data
  for row := 0 to fi.Count -1  do
  begin
    // Split the row by spacces
    lparts := explode(' ',fi[row],0);
    for col := 0 to cols-1 do
    begin
      // Convert current string value to Single
      val := StrToFloat(lparts[col]);
      // Set the correct input or target array value
      if col < nips then data.inp[row,col] := val
      else data.tg[row,col - nips] := val;
    end;
  end;
  fi.Free;
end;

procedure main;
var
  nips, nops, nhid, iterations, i, j: Integer;
  rate, anneal, error: Single;
  NN: TTinyNN;
  data: TTinnData;
  pd: TSingleArray;
  rows: Integer;
begin
  // Important! Initialise the random number generator
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
  BuildData('semeion.data',nips,nops,data);
  rows := High(data.inp);
  // Create the Tinn
  NN := TTinyNN.Create;
  // Prepare Tinn
  NN.Build(nips, nhid, nops);
  // Important! For speed purposes we store the data within the TTinyNN object itself
  NN.SetData(data);
  // Train that brain!
  for i := 0 to iterations-1 do
  begin
    // Shuffle the data on each iteration
    NN.ShuffleData;
    error := 0.00;
    // Train on every row of available data
    for j := 0 to rows -1 do
    begin
      error := error + NN.Train(rate, j);
    end;
    writeln((i+1),' of ',(iterations),' error ',(error/rows):1:10, ' :: learning rate ',rate:1:10);
    // Recalculate rate
    rate := rate * anneal;
  end;
  { Save to a file
    This is slightly different to the original Tinn project in that it saves
    the hidden output layer weights aswell }
  NN.SaveToFile('newtest.tinn');
  // Free the TTinyNN object
  NN.Free;
  // Recreate the TTinyNN object
  NN := TTinyNN.Create;
  // Load our previously saved neural network
  NN.LoadFromFile('newtest.tinn');
  { For testing purposes pass the original training data into the new object, we could however
    at this point load a new data set to run predictions on }
  NN.SetData(data);
  // Pick a random record to perform a prediction on
  i := Random(rows);
  // Perform a prediction
  pd := NN.Predict(i);
  // Dump out the target
  NN.PrintToScreen(data.tg[i], nops);
  // And finally the prediction
  NN.PrintToScreen(pd, nops);
  { If all is well, the prediction that lines up with the target of 1.000000 has
    a value of near to 1 itself }
  NN.Free;
end;

begin
  main;
  {$IFNDEF FPC}
  Readln;
  {$ENDIF}
end.
