using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace Catalyst.Tensors
{
    public delegate object OpHandler(object[] args);

    public static class OpRegistry
    {
        private class OpInstance
        {
            public OpHandler handler;
            public IEnumerable<OpConstraint> constraints;
        }

        private static Dictionary<string, List<OpInstance>> opInstances = new Dictionary<string, List<OpInstance>>();
        // Remember which assemblies have been registered to avoid accidental double-registering
        private static HashSet<Assembly> registeredAssemblies = new HashSet<Assembly>();

        static OpRegistry()
        {
            // Register CPU ops from this assembly
            RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        public static void Register(string opName, OpHandler handler, IEnumerable<OpConstraint> constraints)
        {
            var newInstance = new OpInstance() { handler = handler, constraints = constraints };

            List<OpInstance> instanceList;
            if (opInstances.TryGetValue(opName, out instanceList))
            {
                instanceList.Add(newInstance);
            }
            else
            {
                instanceList = new List<OpInstance>();
                instanceList.Add(newInstance);
                opInstances.Add(opName, instanceList);
            }
        }

        public static object Invoke(string opName, params object[] args)
        {
            List<OpInstance> instanceList;
            if (opInstances.TryGetValue(opName, out instanceList))
            {
                foreach (var instance in instanceList)
                {
                    if (instance.constraints.All(x => x.SatisfiedFor(args)))
                    {
                        return instance.handler.Invoke(args);
                    }
                }

                throw new ApplicationException("None of the registered handlers match the arguments for " + opName);
            }
            else
            {
                throw new ApplicationException("No handlers have been registered for op " + opName);
            }
        }

        public static void RegisterAssembly(Assembly assembly)
        {
            if (!registeredAssemblies.Contains(assembly))
            {
                registeredAssemblies.Add(assembly);

                var types = assembly.TypesWithAttribute<OpsClassAttribute>(false)
                    .Select(x => x.Item1);

                foreach (var type in types)
                {
                    var instance = Activator.CreateInstance(type);

                    var methods = type.MethodsWithAttribute<RegisterOp>(false);
                    foreach (var method in methods)
                    {
                        var paramConstraints = GetParameterConstraints(method.Item1, instance);
                        foreach (var attribute in method.Item2)
                        {
                            attribute.DoRegister(instance, method.Item1, paramConstraints);
                        }
                    }
                }
            }
        }

        private static IEnumerable<OpConstraint> GetParameterConstraints(MethodInfo method, object instance)
        {
            var result = Enumerable.Empty<OpConstraint>();
            foreach (var parameter in method.ParametersWithAttribute<ArgConstraintAttribute>(false))
            {
                foreach (var attribute in parameter.Item2)
                {
                    result = Enumerable.Concat(result, attribute.GetConstraints(parameter.Item1, instance));
                }
            }

            return result;
        }
    }
}
