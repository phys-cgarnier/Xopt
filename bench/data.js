window.BENCHMARK_DATA = {
  "lastUpdate": 1710278758583,
  "repoUrl": "https://github.com/ChristopherMayes/Xopt",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "rroussel@slac.stanford.edu",
            "name": "Ryan Roussel",
            "username": "roussel-ryan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dd04b099baf10f0c171fb6b1ebafed7bd5549fce",
          "message": "Merge pull request #204 from ChristopherMayes/test_functions_and_rasterization\n\nadd test functions and set rasterize for colormaps",
          "timestamp": "2024-03-13T05:51:18+09:00",
          "tree_id": "ca8db834a9b4c7f14daaf483393541ed36b417b6",
          "url": "https://github.com/ChristopherMayes/Xopt/commit/dd04b099baf10f0c171fb6b1ebafed7bd5549fce"
        },
        "date": 1710278683001,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_mo.py::test_mobo[0]",
            "value": 0.3446341976434065,
            "unit": "iter/sec",
            "range": "stddev: 0.5201228981207932",
            "extra": "mean: 2.9016273104583234 sec\nrounds: 24"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[1]",
            "value": 0.33913253501892815,
            "unit": "iter/sec",
            "range": "stddev: 0.6137044531852783",
            "extra": "mean: 2.9486996874074216 sec\nrounds: 27"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[2]",
            "value": 0.3147802458175746,
            "unit": "iter/sec",
            "range": "stddev: 0.6044490508323815",
            "extra": "mean: 3.176819426526316 sec\nrounds: 19"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[3]",
            "value": 0.2974474010314182,
            "unit": "iter/sec",
            "range": "stddev: 0.5480136283883694",
            "extra": "mean: 3.3619389395652304 sec\nrounds: 23"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[4]",
            "value": 0.25968902321668197,
            "unit": "iter/sec",
            "range": "stddev: 0.5929075033354312",
            "extra": "mean: 3.8507596032105287 sec\nrounds: 19"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[5]",
            "value": 0.1881597197722375,
            "unit": "iter/sec",
            "range": "stddev: 1.4032470371328136",
            "extra": "mean: 5.314633765454553 sec\nrounds: 11"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[6]",
            "value": 0.159290503672427,
            "unit": "iter/sec",
            "range": "stddev: 1.546782236363744",
            "extra": "mean: 6.277838144428562 sec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[7]",
            "value": 0.14083124064143102,
            "unit": "iter/sec",
            "range": "stddev: 2.3134877974191266",
            "extra": "mean: 7.100697227727261 sec\nrounds: 11"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[8]",
            "value": 0.13232849439591557,
            "unit": "iter/sec",
            "range": "stddev: 1.9563383923085214",
            "extra": "mean: 7.556951392555599 sec\nrounds: 9"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[9]",
            "value": 0.11953598610625032,
            "unit": "iter/sec",
            "range": "stddev: 1.786337728273908",
            "extra": "mean: 8.365681604125001 sec\nrounds: 8"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[10]",
            "value": 0.11534113134778648,
            "unit": "iter/sec",
            "range": "stddev: 3.0672849372889495",
            "extra": "mean: 8.669934032333307 sec\nrounds: 9"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[11]",
            "value": 0.08784088204824152,
            "unit": "iter/sec",
            "range": "stddev: 2.0282088902856494",
            "extra": "mean: 11.384220839799946 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[12]",
            "value": 0.39882973528502724,
            "unit": "iter/sec",
            "range": "stddev: 0.4413402334911634",
            "extra": "mean: 2.5073356159999984 sec\nrounds: 22"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[13]",
            "value": 0.4196592352908919,
            "unit": "iter/sec",
            "range": "stddev: 0.4388410392753998",
            "extra": "mean: 2.3828857222856965 sec\nrounds: 14"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[14]",
            "value": 0.353240931593656,
            "unit": "iter/sec",
            "range": "stddev: 0.7902610489077367",
            "extra": "mean: 2.8309290078261116 sec\nrounds: 23"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[15]",
            "value": 0.3243308969943447,
            "unit": "iter/sec",
            "range": "stddev: 0.446656154199372",
            "extra": "mean: 3.083270848590897 sec\nrounds: 22"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[16]",
            "value": 0.30186102003256554,
            "unit": "iter/sec",
            "range": "stddev: 0.7518656272370707",
            "extra": "mean: 3.3127828160526236 sec\nrounds: 19"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[17]",
            "value": 0.21666891675807248,
            "unit": "iter/sec",
            "range": "stddev: 1.7175806197569041",
            "extra": "mean: 4.615336684941186 sec\nrounds: 17"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[18]",
            "value": 0.25861768762052945,
            "unit": "iter/sec",
            "range": "stddev: 3.3346753098551014",
            "extra": "mean: 3.8667115509411842 sec\nrounds: 17"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[19]",
            "value": 0.18597117090411336,
            "unit": "iter/sec",
            "range": "stddev: 2.0279648553541674",
            "extra": "mean: 5.377177522399961 sec\nrounds: 10"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[20]",
            "value": 0.14382497322373333,
            "unit": "iter/sec",
            "range": "stddev: 1.9189127538104864",
            "extra": "mean: 6.952895436624942 sec\nrounds: 8"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[21]",
            "value": 0.11977341664602899,
            "unit": "iter/sec",
            "range": "stddev: 3.1749746863252195",
            "extra": "mean: 8.34909805533342 sec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[22]",
            "value": 0.08599760241382373,
            "unit": "iter/sec",
            "range": "stddev: 5.168539085008696",
            "extra": "mean: 11.62823115914281 sec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[23]",
            "value": 0.06766052262730388,
            "unit": "iter/sec",
            "range": "stddev: 5.696891413698217",
            "extra": "mean: 14.779667096400136 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}