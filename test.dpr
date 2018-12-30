program test;

{$IFDEF FPC}
{$mode delphi}{$H+}
{$ELSE}
{$APPTYPE CONSOLE}
{$ENDIF}

uses Classes, SysUtils, pastinn;

procedure BuildData(fn: String; nips: Integer; nops: Integer; var data: TTinnData);
var
  row, rows, col, cols: Integer;
  val: Single;
  lparts: TArray;
  fi: TStrings;
begin
  fi := TStringList.Create;
  fi.LoadFromFile(fn);
  cols := nips + nops;
  rows := fi.Count -1;

  SetLength(data.inp,rows+1,nips);
  SetLength(data.tg,rows+1,nops);
  for row := 0 to fi.Count -1  do
  begin
    lparts := explode(' ',fi[row],0);
    for col := 0 to cols-1 do
    begin
      val := StrToFloat(lparts[col]);
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
  NN.SetData(data);
  // Train that brain!
  for i := 0 to iterations-1 do
  begin
    NN.ShuffleData;
    error := 0.00;
    for j := 0 to rows -1 do
    begin
      error := error + NN.Train(rate, j);
    end;
    writeln((i+1),' of ',(iterations),' error ',(error/rows):1:10, ' :: learning rate ',rate:1:10);
    rate := rate * anneal;
  end;
  { Save to a file
    This is slightly different to the original Tinn project in that it saves
    the hidden output layer weights aswell }
  NN.SaveToFile('newtest.tinn');

  NN.Free;
  NN := TTinyNN.Create;

  NN.LoadFromFile('newtest.tinn');
  NN.SetData(data);

  { Perform a prediction, ideally a test set would be loaded to make the prediction
    with, but for testing purposes we are just reusing the training set loaded
    earlier. One data set is picked at random - as the data was shuffled earlier
    we can just use the first index of the input and target arrays }
  pd := NN.Predict(0);
  // Dump out the target
  NN.PrintToScreen(data.tg[0], nops);
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
