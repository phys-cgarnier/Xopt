window.BENCHMARK_DATA = {
  "lastUpdate": 1706643813121,
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
          "id": "dec6a484c564049c10a51e93934f6ad4323c0c2c",
          "message": "Merge pull request #187 from kathryn-baker/main\n\nResolving TypeError caused by using GPU and max_travel_distances in Generator",
          "timestamp": "2024-01-30T13:04:34-06:00",
          "tree_id": "81c2ffa41c535d6d18ff3244c3dfd1f59b97d288",
          "url": "https://github.com/ChristopherMayes/Xopt/commit/dec6a484c564049c10a51e93934f6ad4323c0c2c"
        },
        "date": 1706643756146,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[0]",
            "value": 0.3180928620751957,
            "unit": "iter/sec",
            "range": "stddev: 1.1498205984035015",
            "extra": "mean: 3.1437360570624957 sec\nrounds: 16"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[1]",
            "value": 0.32918475577943296,
            "unit": "iter/sec",
            "range": "stddev: 0.5903242817042211",
            "extra": "mean: 3.0378077430476162 sec\nrounds: 21"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[2]",
            "value": 0.288377359377299,
            "unit": "iter/sec",
            "range": "stddev: 0.6303718504136323",
            "extra": "mean: 3.4676786075000026 sec\nrounds: 20"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[3]",
            "value": 0.2735471755163602,
            "unit": "iter/sec",
            "range": "stddev: 0.4839026963319662",
            "extra": "mean: 3.655676568812506 sec\nrounds: 16"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[4]",
            "value": 0.2479268272181555,
            "unit": "iter/sec",
            "range": "stddev: 0.8126168348735895",
            "extra": "mean: 4.0334481396 sec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[5]",
            "value": 0.17226243513557257,
            "unit": "iter/sec",
            "range": "stddev: 1.5782005829160046",
            "extra": "mean: 5.805096155833326 sec\nrounds: 12"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[6]",
            "value": 0.16544339230731545,
            "unit": "iter/sec",
            "range": "stddev: 0.45934269703847036",
            "extra": "mean: 6.044363489249989 sec\nrounds: 8"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[7]",
            "value": 0.12902857633606446,
            "unit": "iter/sec",
            "range": "stddev: 1.9997639663879556",
            "extra": "mean: 7.7502211401250065 sec\nrounds: 8"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[8]",
            "value": 0.14162200965656677,
            "unit": "iter/sec",
            "range": "stddev: 1.5271724420842205",
            "extra": "mean: 7.061049355428573 sec\nrounds: 7"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[9]",
            "value": 0.12114579145003111,
            "unit": "iter/sec",
            "range": "stddev: 2.3012633557481994",
            "extra": "mean: 8.254517041249997 sec\nrounds: 8"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[10]",
            "value": 0.10353452084981901,
            "unit": "iter/sec",
            "range": "stddev: 2.3784854954425505",
            "extra": "mean: 9.658614264999983 sec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[11]",
            "value": 0.06391662887964061,
            "unit": "iter/sec",
            "range": "stddev: 4.84656217102492",
            "extra": "mean: 15.64538082699994 sec\nrounds: 4"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[12]",
            "value": 0.36756222833656466,
            "unit": "iter/sec",
            "range": "stddev: 0.4697298603981364",
            "extra": "mean: 2.7206277547222095 sec\nrounds: 18"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[13]",
            "value": 0.42384424166321455,
            "unit": "iter/sec",
            "range": "stddev: 0.337261132140455",
            "extra": "mean: 2.3593572867142956 sec\nrounds: 28"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[14]",
            "value": 0.36134077872517056,
            "unit": "iter/sec",
            "range": "stddev: 0.6010516811794013",
            "extra": "mean: 2.767470650636369 sec\nrounds: 22"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[15]",
            "value": 0.28438276768023946,
            "unit": "iter/sec",
            "range": "stddev: 1.0848077746979894",
            "extra": "mean: 3.516387466642852 sec\nrounds: 14"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[16]",
            "value": 0.27644118889732205,
            "unit": "iter/sec",
            "range": "stddev: 0.5622312617747125",
            "extra": "mean: 3.6174059444210678 sec\nrounds: 19"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[17]",
            "value": 0.21834423006071282,
            "unit": "iter/sec",
            "range": "stddev: 0.7507592919837152",
            "extra": "mean: 4.579924093812508 sec\nrounds: 16"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[18]",
            "value": 0.19350175007336096,
            "unit": "iter/sec",
            "range": "stddev: 3.5351653796417977",
            "extra": "mean: 5.167911916149993 sec\nrounds: 60"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[19]",
            "value": 0.16882204043949364,
            "unit": "iter/sec",
            "range": "stddev: 2.5128992262883263",
            "extra": "mean: 5.9233971902999425 sec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[20]",
            "value": 0.17524673923679707,
            "unit": "iter/sec",
            "range": "stddev: 2.5506853492685115",
            "extra": "mean: 5.706240266466693 sec\nrounds: 15"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[21]",
            "value": 0.08023563014403146,
            "unit": "iter/sec",
            "range": "stddev: 5.500587232178619",
            "extra": "mean: 12.463290912090976 sec\nrounds: 11"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[22]",
            "value": 0.09231506900005862,
            "unit": "iter/sec",
            "range": "stddev: 2.4395888783852864",
            "extra": "mean: 10.832467665700005 sec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[23]",
            "value": 0.07263749697681447,
            "unit": "iter/sec",
            "range": "stddev: 4.30919617711434",
            "extra": "mean: 13.766994205750166 sec\nrounds: 4"
          }
        ]
      }
    ]
  }
}