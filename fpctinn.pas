unit fpctinn;

{$mode delphi}{$H+}

interface

uses Classes, SysUtils;

type
  TSingleArray = array of Single;
  TfpcTinn = record
    // All the weights.
    w: TSingleArray;
    // Hidden to output layer weights.
    x: TSingleArray;
    // Biases.
    b: TSingleArray;
    // Hidden layer.
    h: TSingleArray;
    // Output layer.
    o: TSingleArray;
    // Number of biases - always two - Tinn only supports a single hidden layer.
    nb: Integer;
    // Number of weights.
    nw: Integer;
    // Number of inputs.
    nips: Integer;
    // Number of hidden neurons.
    nhid: Integer;
    // Number of outputs.
    nops: Integer;
  end;

function xtpredict(var tinn: TFpcTinn; inp: TSingleArray): TSingleArray;
function xttrain(var tinn: TfpcTinn; inp: TSingleArray; tg: TSingleArray; rate: Single): Single;
function xtbuild(nips: Integer; nhid: Integer; nops: Integer): TfpcTinn;
procedure xtsave(tinn: TfpcTinn; path: String);
function xtload(path: String): TfpcTinn;
procedure xtprint(arr: TSingleArray; size: Integer);

implementation

// Computes error
function err(a: Single; b: Single): Single;
begin
  Result := 0.5 * (a - b) * (a - b);
end;

// Returns partial derivative of error function.
function pderr(a: Single; b: Single): Single;
begin
  Result := a - b;
end;

// Computes total error of target to output.
function toterr(tg: TSingleArray; o: TSingleArray; size: Integer): Single;
var
  i: Integer;
begin
  Result := 0.00;
  for i := 0 to size-1 do
  begin
    Result := Result + err(tg[i], o[i]);
  end;
end;

// Activation function.
function act(a: Single): Single;
begin
  Result := 1.0 / (1.0 + exp(-a));
end;

// Returns partial derivative of activation function.
function pdact(a: Single): Single;
begin
  Result := a * (1.0 - a);
end;

// Performs back propagation
procedure bprop(var t: TfpcTinn; inp: TSingleArray; tg: TSingleArray; rate: Single);
var
  i,j: Integer;
  a,b,sum: Single;
begin
  for i := 0 to t.nhid-1 do
  begin
    sum := 0.00;
    // Calculate total error change with respect to output
    for j := 0 to t.nops-1 do
    begin
      a := pderr(t.o[j], tg[j]);
      b := pdact(t.o[j]);
      sum += a * b * t.x[j * t.nhid + i];
      // Correct weights in hidden to output layer
      t.x[j * t.nhid + i] -= rate * a * b * t.h[i];
    end;
    // Correct weights in input to hidden layer
    for j := 0 to t.nips-1 do
    begin
      t.w[i * t.nips + j] -= rate * sum * pdact(t.h[i]) * inp[j];
    end;
  end;
end;

// Performs forward propagation
procedure fprop(var t: TfpcTinn; inp: TSingleArray);
var
  i,j: Integer;
  sum: Single;
begin
  // Calculate hidden layer neuron values
  for i := 0 to t.nhid-1 do
  begin
    sum := 0.00;
    for j := 0 to t.nips-1 do
    begin
      sum += inp[j] * t.w[i * t.nips + j];
    end;
    t.h[i] := act(sum + t.b[0]);
  end;
  // Calculate output layer neuron values
  for i := 0 to t.nops-1 do
  begin
    sum := 0.00;
    for j := 0 to t.nhid-1 do
    begin
      sum += t.h[j] * t.x[i * t.nhid + j];
    end;
    t.o[i] := act(sum + t.b[1]);
  end;
end;

// Randomizes tinn weights and biases
procedure wbrand(var t: TfpcTinn);
var
  i: Integer;
begin
  for i := 0 to t.nw-1 do t.w[i] := Random - 0.5;
  for i := 0 to t.nb-1 do t.b[i] := Random - 0.5;
end;

// Returns an output prediction given an input
function xtpredict(var tinn: TFpcTinn; inp: TSingleArray): TSingleArray;
begin
  fprop(tinn, inp);
  Result := tinn.o;
end;

// Trains a tinn with an input and target output with a learning rate. Returns target to output error
function xttrain(var tinn: TfpcTinn; inp: TSingleArray; tg: TSingleArray; rate: Single): Single;
begin
  fprop(tinn, inp);
  bprop(tinn, inp, tg, rate);
  Result := toterr(tg, tinn.o, tinn.nops);
end;

function xtbuild(nips: Integer; nhid: Integer; nops: Integer): TfpcTinn;
begin
  Result.nb := 2;
  Result.nw := nhid * (nips + nops);
  SetLength(Result.w,Result.nw);
  SetLength(Result.x,(High(Result.w) + nhid * nips));
  SetLength(Result.b,Result.nb);
  SetLength(Result.h,nhid);
  SetLength(Result.o,nops);
  Result.nips := nips;
  Result.nhid := nhid;
  Result.nops := nops;
  wbrand(Result);
end;

procedure xtsave(tinn: TfpcTinn; path: String);
var
  F: TextFile;
  i: Integer;
begin
  AssignFile(F,path);
  Rewrite(F);
  writeln(F,tinn.nips,' ',tinn.nhid,' ',tinn.nops);
  for i := 0 to tinn.nb-1 do
  begin
    writeln(F,tinn.b[i]:1:6);
  end;
  for i := 0 to tinn.nw-1 do
  begin
    writeln(F,tinn.w[i]:1:6);
  end;
  CloseFile(F);
end;

function xtload(path: String): TfpcTinn;
var
  F: TextFile;
  i, nips, nhid, nops: Integer;
  //l: String;
  l: Single;
  s: String;
begin
  AssignFile(F,path);
  Reset(F);
  nips := 0;
  nhid := 0;
  nops := 0;
  // Read header
  Readln(F,s);
  sscanf(s,'%d %d %d',[@nips, @nhid, @nops]);
  Result := xtbuild(nips, nhid, nops);
  for i := 0 to Result.nb-1 do
  begin
    Readln(F,l);
    //Result.b[i] := StrToFloat(l);
    Result.b[i] := l;
  end;
  for i := 0 to Result.nw-1 do
  begin
    Readln(F,l);
    //Result.w[i] := StrToFloat(l);
    Result.w[i] := l;
  end;
  CloseFile(F);
end;

procedure xtprint(arr: TSingleArray; size: Integer);
var
  i: Integer;
begin
  for i := 0 to size-1 do
  begin
    write(arr[i]:1:8,' ');
  end;
  writeln;
end;

end.
