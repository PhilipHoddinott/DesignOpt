classdef pNNAtt
    %PNNATT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Property1;
         TimeProp = datestr(now);
        HL_In;
        HL_Vec;
        HL_W;
        HL_D;
        
         
    end
    %{
    methods
        function obj = pNNAtt(inputArg1,inputArg2)
            %PNNATT Construct an instance of this class
            %   Detailed explanation goes here
            obj.Property1 = inputArg1 + inputArg2;
        end
        
        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
    %}
end

