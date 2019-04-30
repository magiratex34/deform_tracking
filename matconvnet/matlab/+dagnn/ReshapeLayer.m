% Wrapper for Reshape layer:


classdef ReshapeLayer < dagnn.Layer
  properties
      dims
  end
  methods
    function outputs = forward(obj, inputs, params)
        sz = size(inputs{1});
        assert(prod(sz) == obj.dims(1)*obj.dims(2));
        
        outputs = vl_nnreshape(inputs{1}, obj.dims, []);
%       outputs = vl_nnbilinearsampler(inputs{1}, inputs{2});
      outputs = {outputs};
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
%       [dX,dG] = vl_nnbilinearsampler(inputs{1}, inputs{2}, derOutputs{1});
%       derInputs = {dX,dG};
%       derParams = {};
        
        dy = vl_nnreshape(inputs{1}, obj.dims, derOutputs{1});
        derInputs = {dy};
        derParams = {};
    end

    function obj = ReshapeLayer(varargin)
%         obj.load(varargin);
        if nargin == 0
            obj.dims = [0 0];
        else
            obj.dims = varargin{1};
        end;
    end
  end
end
