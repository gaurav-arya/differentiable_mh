using Pkg

using Documenter, DifferentiableMH

### Formatting

# Raw CSS file based on DocThemeIndigo
indigo = "assets/indigo.css" 
format = Documenter.HTML(prettyurls = false,
    assets = [indigo, "assets/extra_styles.css"],
    repolink = "https://github.com/gaurav-arya/DifferentiableMH.jl",
    edit_link = "main")

### Pagination

pages = [
    "Overview" => "index.md",
    "Tutorials" => [
        "tutorials/analyze_gaussian_mh_problem.md",
        "tutorials/analyze_ising_problem.md",
        "tutorials/analyze_mh_tuning_problem.md",
        "tutorials/analyze_prior_sensitivity_problem.md",
    ],
    "Public API" => "public_api.md"
]

### Make docs

makedocs(sitename = "DifferentiableMH.jl",
    authors = "Gaurav Arya and other contributors",
    modules = [DifferentiableMH],
    format = format,
    pages = pages,
    warnonly = [:missing_docs])

println("Doc deployment disabled.")
# try
#     deploydocs(repo = "github.com/gaurav-arya/DifferentiableMH.jl",
#         devbranch = "main",
#         push_preview = true)
# catch e
#     println("Error encountered while deploying docs:")
#     showerror(stdout, e)
# end
