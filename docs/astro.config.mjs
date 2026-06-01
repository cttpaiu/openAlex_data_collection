// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  site: "https://darunesh1.github.io",
  base: "/openAlex_data_collection/",
  integrations: [
    starlight({
      title: "openalex-data-collection",
      description:
        "CLI pipeline for collecting quantum-computing papers from OpenAlex into DuckDB.",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/Darunesh1/openAlex_data_collection",
        },
      ],
      sidebar: [
        {
          label: "User Guide",
          items: [
            { label: "Overview", slug: "user/overview" },
            { label: "Installation", slug: "user/installation" },
            { label: "Quick Start", slug: "user/quick-start" },
            { label: "Configuration", slug: "user/configuration" },
            { label: "Workflow", slug: "user/workflow" },
            {
              label: "Commands",
              collapsed: true,
              items: [
                { label: "init", slug: "user/commands/init" },
                { label: "validate", slug: "user/commands/validate" },
                {
                  label: "search / search-filtered",
                  slug: "user/commands/search",
                },
                { label: "get-topics", slug: "user/commands/get-topics" },
                { label: "check-anchor", slug: "user/commands/check-anchor" },
                { label: "sample", slug: "user/commands/sample" },
                { label: "download", slug: "user/commands/download" },
                { label: "convert-to-db", slug: "user/commands/convert-to-db" },
                { label: "check-db", slug: "user/commands/check-db" },
                { label: "import-wos", slug: "user/commands/import-wos" },
                { label: "import-wos-csv", slug: "user/commands/import-wos-csv" },
                {
                  label: "impute crossref",
                  slug: "user/commands/impute-crossref",
                },
                { label: "impute llm", slug: "user/commands/impute-llm" },
                { label: "impute pdf", slug: "user/commands/impute-pdf" },
              ],
            },
          ],
        },
        {
          label: "Developer Guide",
          items: [
            { label: "Architecture", slug: "developer/architecture" },
            { label: "Pipeline data flow", slug: "developer/pipeline" },
            { label: "DuckDB schema", slug: "developer/schema" },
            { label: "Dependencies", slug: "developer/dependencies" },
            { label: "Testing", slug: "developer/testing" },
            { label: "Contributing", slug: "developer/contributing" },
            {
              label: "Modules",
              collapsed: true,
              items: [
                { label: "openalex/cli.py", slug: "developer/modules/cli" },
                { label: "openalex/config.py", slug: "developer/modules/config" },
                {
                  label: "openalex/api_client.py",
                  slug: "developer/modules/api-client",
                },
                { label: "openalex/utils.py", slug: "developer/modules/utils" },
                {
                  label: "openalex/validator.py",
                  slug: "developer/modules/validator",
                },
                {
                  label: "openalex/imputation.py",
                  slug: "developer/modules/imputation",
                },
                { label: "openalex/pdf.py", slug: "developer/modules/pdf" },
                { label: "openalex/wos.py", slug: "developer/modules/wos" },
                {
                  label: "openalex/anchors.py",
                  slug: "developer/modules/anchors",
                },
              ],
            },
            {
              label: "Command internals",
              collapsed: true,
              items: [
                { label: "download", slug: "developer/commands/download" },
                { label: "convert-to-db", slug: "developer/commands/database" },
                { label: "check-db", slug: "developer/commands/check-db" },
                {
                  label: "impute-affiliation (llm)",
                  slug: "developer/commands/impute-affiliation",
                },
                {
                  label: "enrich-crossref",
                  slug: "developer/commands/enrich-crossref",
                },
                { label: "impute-pdf", slug: "developer/commands/impute-pdf" },
                { label: "import-wos", slug: "developer/commands/import-wos" },
                { label: "import-wos-csv", slug: "developer/commands/import-wos-csv" },
              ],
            },
          ],
        },
      ],
    }),
  ],
});
