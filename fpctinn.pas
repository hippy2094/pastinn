unit fpctinn;

{$mode delphi}{$H+}

interface

uses Classes, SysUtils;

type
  TSingleArray = array of Single;
  TfpcTinn = record
    w: TSingleArray; // All the weights    
    x: TSingleArray; // Hidden to output layer weights    
    b: TSingleArray; // Biases    
    h: TSingleArray; // Hidden layer    
    o: TSingleArray; // Output layer    
    nb: Integer; // Number of biases - always two - Tinn only supports a single hidden layer.
    nw: Integer; // Number of weights.
    nips: Integer; // Number of inputs.    
    nhid: Integer; // Number of hidden neurons.    
    nops: Integer; // Number of outputs.
  end;
  TTinyNN = class(TObject)
    private
      FTinn: TFpcTinn;
      function err(a: Single; b: Single): Single;
      function pderr(a: Single; b: Single): Single;
      function toterr(tg: TSingleArray; o: TSingleArray; size: Integer): Single;
      function act(a: Single): Single;
      function pdact(a: Single): Single;
      procedure bprop(inp: TSingleArray; tg: TSingleArray; rate: Single);
      procedure fprop(inp: TSingleArray);
      procedure wbrand;
    public
      function Train(inp: TSingleArray; tg: TSingleArray; rate: Single): Single;
      procedure Build(nips: Integer; nhid: Integer; nops: Integer);
      function Predict(inp: TSingleArray): TSingleArray;
      procedure SaveToFile(path: String);
      procedure LoadFromFile(path: String);
      procedure PrintToScreen(arr: TSingleArray; size: Integer);
  end;

implementation

{ TTinyNN }

// Computes error
function TTinyNN.err(a: Single; b: Single): Single;
begin
  Result := 0.5 * (a - b) * (a - b);
end;

// Returns partial derivative of error function.
function TTinyNN.pderr(a: Single; b: Single): Single;
begin
  Result := a - b;
end;

// Computes total error of target to output.
function TTinyNN.toterr(tg: TSingleArray; o: TSingleArray; size: Integer): Single;
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
function TTinyNN.act(a: Single): Single;
begin
  Result := 1.0 / (1.0 + exp(-a));
end;

// Returns partial derivative of activation function.
function TTinyNN.pdact(a: Single): Single;
begin
  Result := a * (1.0 - a);
end;

// Performs back propagation
procedure TTinyNN.bprop(inp: TSingleArray; tg: TSingleArray; rate: Single);
var
  i,j: Integer;
  a,b,sum: Single;
begin
  for i := 0 to FTinn.nhid-1 do
  begin
    sum := 0.00;
    // Calculate total error change with respect to output
    for j := 0 to FTinn.nops-1 do
    begin
      a := pderr(FTinn.o[j], tg[j]);
      b := pdact(FTinn.o[j]);
      sum += a * b * FTinn.x[j * FTinn.nhid + i];
      // Correct weights in hidden to output layer
      FTinn.x[j * FTinn.nhid + i] -= rate * a * b * FTinn.h[i];
    end;
    // Correct weights in input to hidden layer
    for j := 0 to FTinn.nips-1 do
    begin
      FTinn.w[i * FTinn.nips + j] -= rate * sum * pdact(FTinn.h[i]) * inp[j];
    end;
  end;
end;

// Performs forward propagation
procedure TTinyNN.fprop(inp: TSingleArray);
var
  i,j: Integer;
  sum: Single;
begin
  // Calculate hidden layer neuron values
  for i := 0 to FTinn.nhid-1 do
  begin
    sum := 0.00;
    for j := 0 to FTinn.nips-1 do
    begin
      sum += inp[j] * FTinn.w[i * FTinn.nhid + j];
    end;
    FTinn.h[i] := act(sum + FTinn.b[0]);
  end;
  // Calculate output layer neuron values
  for i := 0 to FTinn.nops-1 do
  begin
    sum := 0.00;
    for j := 0 to FTinn.nhid-1 do
    begin
      sum += FTinn.h[j] * FTinn.x[i * FTinn.nhid + j];
    end;
    FTinn.o[i] := act(sum + FTinn.b[1]);
  end;
end;

// Randomizes tinn weights and biases
procedure TTinyNN.wbrand;
var
  i: Integer;
begin
  for i := 0 to FTinn.nw-1 do FTinn.w[i] := Random - 0.5;
  for i := 0 to FTinn.nb-1 do FTinn.b[i] := Random - 0.5;
end;

// Returns an output prediction given an input
function TTinyNN.Predict(inp: TSingleArray): TSingleArray;
begin
  fprop(inp);
  Result := FTinn.o;
end;

// Trains a tinn with an input and target output with a learning rate. Returns target to output error
function TTinyNN.Train(inp: TSingleArray; tg: TSingleArray; rate: Single): Single;
begin
  fprop(inp);
  bprop(inp, tg, rate);
  Result := toterr(tg, FTinn.o, FTinn.nops);
end;

procedure TTinyNN.Build(nips: Integer; nhid: Integer; nops: Integer);
begin
  FTinn.nb := 2;
  FTinn.nw := nhid * (nips + nops);
  SetLength(FTinn.w,FTinn.nw);
  SetLength(FTinn.x,(High(FTinn.w) + nhid * nips));
  SetLength(FTinn.b,FTinn.nb);
  SetLength(FTinn.h,nhid);
  SetLength(FTinn.o,nops);
  FTinn.nips := nips;
  FTinn.nhid := nhid;
  FTinn.nops := nops;
  wbrand;
end;

procedure TTinyNN.SaveToFile(path: String);
var
  F: TextFile;
  i: Integer;
begin
  AssignFile(F,path);
  Rewrite(F);
  writeln(F,FTinn.nips,' ',FTinn.nhid,' ',FTinn.nops);
  for i := 0 to FTinn.nb-1 do
  begin
    writeln(F,FTinn.b[i]{:1:6});
  end;
  for i := 0 to FTinn.nw-1 do
  begin
    writeln(F,FTinn.w[i]{:1:6});
  end;
  CloseFile(F);
end;

procedure TTinyNN.LoadFromFile(path: String);
var
  F: TextFile;
  i, nips, nhid, nops: Integer;
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
  Build(nips, nhid, nops);
  for i := 0 to FTinn.nb-1 do
  begin
    Readln(F,l);
    FTinn.b[i] := l;
  end;
  for i := 0 to FTinn.nw-1 do
  begin
    Readln(F,l);
    FTinn.w[i] := l;
  end;
  CloseFile(F);
end;

procedure TTinyNN.PrintToScreen(arr: TSingleArray; size: Integer);
var
  i: Integer;
begin
  for i := 0 to size-1 do
  begin
    write(arr[i]:1:6,' ');
  end;
  writeln;
end;

end.
