using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace Catalyst.Tensors
{
    [AttributeUsage(AttributeTargets.Class)]
    public class OpsClassAttribute : Attribute
    {
    }

    [AttributeUsage(AttributeTargets.Method)]
    public abstract class RegisterOp : Attribute
    {
        public string OpName { get; private set; }

        public RegisterOp(string opName)
        {
            OpName = opName;
        }

        public abstract void DoRegister(object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints);
    }

    /// <summary>
    /// Register a method where the only constraint is that the argument counts match.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class RegisterOpArgCount : RegisterOp
    {
        public RegisterOpArgCount(string opName) : base(opName)
        {
        }

        public override void DoRegister(object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints)
        {
            var constraints = new List<OpConstraint>();
            constraints.AddRange(paramConstraints);
            constraints.Add(new ArgCountConstraint(method.GetParameters().Length));

            OpRegistry.Register(OpName, args => method.Invoke(instance, args), constraints);
        }
    }

    
    [AttributeUsage(AttributeTargets.Method)]
    public class RegisterOpStorageType : RegisterOp
    {
        private readonly Type storageType;

        public RegisterOpStorageType(string opName, Type storageType) : base(opName)
        {
            this.storageType = storageType;
        }

        public override void DoRegister(object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints)
        {
            var constraints = new List<OpConstraint>();
            constraints.AddRange(paramConstraints);
            constraints.Add(new ArgCountConstraint(method.GetParameters().Length));

            var methodParams = method.GetParameters();
            for(int i = 0; i < methodParams.Length; ++i)
            {
                if (methodParams[i].ParameterType == typeof(Tensor))
                {
                    constraints.Add(new ArgStorageTypeConstraint(i, storageType));
                }
            }

            OpRegistry.Register(OpName, args => method.Invoke(instance, args), constraints);
        }
    }




    [AttributeUsage(AttributeTargets.Parameter)]
    public abstract class ArgConstraintAttribute : Attribute
    {
        public ArgConstraintAttribute()
        {
        }

        public abstract IEnumerable<OpConstraint> GetConstraints(ParameterInfo parameter, object instance);
    }

    [AttributeUsage(AttributeTargets.Parameter)]
    public class OpArgStorageType : ArgConstraintAttribute
    {
        private readonly Type storageType;

        public OpArgStorageType(Type storageType)
        {
            this.storageType = storageType;
        }

        public override IEnumerable<OpConstraint> GetConstraints(ParameterInfo parameter, object instance)
        {
            yield return new ArgStorageTypeConstraint(parameter.Position, storageType);
        }
    }
}
