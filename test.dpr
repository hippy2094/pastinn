program test;

{$IFDEF FPC}
{$mode delphi}{$H+}
{$ELSE}
{$APPTYPE CONSOLE}
{$ENDIF}

uses Classes, SysUtils, pastinn;

type
  TArray = array of string;
  TTestData = record
    inp: array of TSingleArray;
    tg: array of TSingleArray;
    nips: Integer;
    nops: Integer;
    rows: Integer;
  end;

function explode(cDelimiter,  sValue : string; iCount : integer) : TArray;
var
  s : string;
  i,p : integer;
begin
  s := sValue; i := 0;
  while length(s) > 0 do
  begin
    inc(i);
    SetLength(result, i);
    p := pos(cDelimiter,s);
    if ( p > 0 ) and ( ( i < iCount ) OR ( iCount = 0) ) then
    begin
      result[i - 1] := copy(s,0,p-1);
      s := copy(s,p + length(cDelimiter),length(s));
    end else
    begin
      result[i - 1] := s;
      s :=  '';
    end;
  end;
end;

function InitData(nips: Integer; nops: Integer; rows: Integer): TTestData;
begin
  SetLength(Result.inp,rows,nips);
  SetLength(Result.tg,rows,nops);
  Result.nips := nips;
  Result.nops := nops;
  Result.rows := rows;
end;

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

procedure main;
var
  nips, nops, nhid, iterations, i, j: Integer;
  rate, anneal, error: Single;
  data: TTestData;
  NN: TTinyNN;
  pd: TSingleArray;
begin
  Randomize;
  nips := 256;
  nops := 10;
  rate := 1.00;
  nhid := 28;
  anneal := 0.99;
  iterations := 128;
  data := build(nips, nops);
  NN := TTinyNN.Create;
  NN.Build(nips, nhid, nops);
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
  NN.SaveToFile('test.tinn');
{  shuffle(data);
  NN.LoadFromFile('test.tinn');}
  pd := NN.Predict(data.inp[0]);
  NN.PrintToScreen(data.tg[0], data.nops);
  NN.PrintToScreen(pd, data.nops);
  NN.Free;
end;

begin
  main;
  Readln;
end.
