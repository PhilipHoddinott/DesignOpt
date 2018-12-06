classdef testCls
    %TESTCLS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        P1;
        P2;
        P3;
        P4;
    end
    
    methods
        function obj = testCls(inputArg1,inputArg2)
            %TESTCLS Construct an instance of this class
            %   Detailed explanation goes here
            obj.P1 = inputArg1;
            obj.P2=inputArg2;
            obj.P3=obj.P1+obj.P2;
            obj.P4=zeros(obj.P3,obj.P2);
        end
        
        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end

