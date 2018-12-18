unit fpctinn;

{$mode objfpc}{$H+}

interface

uses Classes, SysUtils;

type
  TSingleArray = array of Single;
  TfpcTinn = record
    // All the weights.
    w: array of Single;
    // Hidden to output layer weights.
    x: array of Single;
    // Biases.
    b: array of Single;
    // Hidden layer.
    h: array of Single;
    // Output layer.
    o: array of Single;
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
  
function xtpredict(tinn: TFpcTinn; inp: TSingleArray): TSingleArray;
function xttrain(tinn: TfpcTinn; inp: TSingleArray; tg: TSingleArray; rate: Single): Single;
function xtbuild(nips: Integer; nhid: Integer; nops: Integer): TfpcTinn;
function xtsave(tinn: TfpcTinn; path: String): Boolean;
function xtload(path: String): TfpcTinn;
procedure xtfree(var tinn: TfpcTinn);
procedure xtprint(arr: TSingleArray; size: Integer);
  
implementation

function xtpredict(tinn: TFpcTinn; inp: TSingleArray): TSingleArray;
begin
end;

function xttrain(tinn: TfpcTinn; inp: TSingleArray; tg: TSingleArray; rate: Single): Single;
begin
end;

function xtbuild(nips: Integer; nhid: Integer; nops: Integer): TfpcTinn;
begin
end;

function xtsave(tinn: TfpcTinn; path: String): Boolean;
begin
end;

function xtload(path: String): TfpcTinn;
begin
end;

procedure xtfree(var tinn: TfpcTinn);
begin
end;

procedure xtprint(arr: TSingleArray; size: Integer);
begin
end;

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
  for i := 0 to size do
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

// Returns floating point random from 0.0 - 1.0.
function frand: Single;
begin
  Result := Random;
end; 

end.
