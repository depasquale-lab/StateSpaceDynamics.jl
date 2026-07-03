# Building the Documentation

The docs are built with [Documenter.jl](https://documenter.juliadocs.org/stable/).
Tutorial pages under `docs/src/tutorials/` are **generated** from the
[Literate.jl](https://fredrikekre.github.io/Literate.jl/) scripts in
`docs/examples/` — edit those `.jl` files, never the generated Markdown.

## One-time setup

From the repository root, instantiate the docs environment and `dev` your local
copy of the package into it:

```julia
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

## Building

Build the site (converts the Literate examples, runs every tutorial, and writes
HTML to `docs/build/`):

```julia
julia --project=docs docs/make.jl
```

Note that the build executes all tutorial code, so it takes a few minutes.

## Viewing locally

Documenter's output needs to be served over HTTP for links and search to work
(see the note at the end of Documenter's ["Building an Empty
Document"](https://documenter.juliadocs.org/stable/man/guide/) section). In a
second Julia process, run:

```julia
julia --project=docs docs/server.jl
```

and open the localhost URL it prints.

## Tips

- The example scripts double as tests: `Pkg.test()` runs every file in
  `docs/examples/` (lines ending in `#src` are test-only and are stripped from
  the rendered tutorials).
- Docstrings live in `src/`; after editing them, re-run `make.jl` to see the
  changes.
- When adding a new tutorial, register it in the `tutorials` list *and* the
  `pages` section of `docs/make.jl`.
