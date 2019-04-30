classdef Clip < dagnn.ElementWise
  properties
    thres = 1e-15
  end

  properties (Transient)
  end

  methods
    function outputs = forward(obj, inputs, params)
        
      if inputs{1}<obj.thres
          p = obj.thres;
      elseif inputs{1}>1-obj.thres
          p = 1;
      else
          p = inputs{1};
      end;
      outputs{1} = p;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      dy = derOutputs{1};
      
      if inputs{1}<obj.thres || inputs{1}>1-obj.thres
          derInputs{1} = obj.thres * dy;
      else
          derInputs{1} = 1 * dy;
      end;
      derParams = {};
      
    end

    % ---------------------------------------------------------------------
    function obj = Clip(varargin)
      obj.load(varargin{:}) ;
    end

    function obj = reset(obj)
      reset@dagnn.ElementWise(obj) ;
      
    end
  end
end
