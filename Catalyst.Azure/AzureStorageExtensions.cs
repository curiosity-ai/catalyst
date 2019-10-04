using System;

using Catalyst.Azure;

namespace Microsoft.Extensions.DependencyInjection
{
    public static class AzureStorageExtensions
    {
        public static void AddAzureStorageContext(this IServiceCollection services, Action<AzureStorageSettings> setupAction) 
        {
            if (setupAction == null) {
                throw new ArgumentNullException(nameof(setupAction));
            }
            services.AddSingleton<AzureStorageContext>((serviceProvider) => {
                AzureStorageSettings settings = new AzureStorageSettings();
                setupAction(settings);
                return new DefaultAzureStorageContext(settings.ConnectionString);
            });
        }

        public static void AddAzureStorageContext<TAzureStorageContext>(this IServiceCollection services, Action<AzureStorageSettings> setupAction)
                    where TAzureStorageContext : AzureStorageContext
        {
            if (setupAction == null) {
                throw new ArgumentNullException(nameof(setupAction));
            }

            services.AddSingleton<TAzureStorageContext>((serviceProvider) => {
                AzureStorageSettings settings = new AzureStorageSettings();
                setupAction(settings);
                return (TAzureStorageContext)Activator.CreateInstance(typeof(TAzureStorageContext), new object[] { settings.ConnectionString });
            });
        }
    }
}
