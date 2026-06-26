from __future__ import annotations

import time
from collections import Counter
from math import ceil

from ortools.sat.python import cp_model

from . import model as _model
from .schema import (
    AllocationBus,
    AllocationResult,
    BusConfiguration,
    OptimizationInput,
    Passenger,
    ObjectiveValue,
)
from .validation import validate_input, validate_result

_original_optimize = _model.optimize


def _has_single_choice_passengers(data: OptimizationInput) -> bool:
    return any(
        passenger.first_choice == passenger.second_choice
        for passenger in data.passengers
    )


def _solve_single_choice_primary_objectives(
    data: OptimizationInput,
    passengers: tuple[Passenger, ...],
    destinations: tuple[str, ...],
    progress,
    cancellation_check,
    deadline: float,
):
    model = cp_model.CpModel()
    pair_counts = Counter(
        (passenger.first_choice, passenger.second_choice)
        for passenger in passengers
    )
    assigned_by_pair_and_destination = {}
    first_choice_demand = Counter(passenger.first_choice for passenger in passengers)
    fixed_first_choice_demand = Counter(
        passenger.first_choice
        for passenger in passengers
        if passenger.first_choice == passenger.second_choice
    )
    flexible_first_choice_demand = Counter(
        passenger.first_choice
        for passenger in passengers
        if passenger.first_choice != passenger.second_choice
    )

    for pair_index, (pair, count) in enumerate(sorted(pair_counts.items())):
        first, second = pair
        if first == second:
            fixed_count = model.NewIntVar(count, count, f"pair_{pair_index}_single")
            assigned_by_pair_and_destination[(pair, first)] = fixed_count
            continue

        first_count = model.NewIntVar(0, count, f"pair_{pair_index}_first")
        second_count = model.NewIntVar(0, count, f"pair_{pair_index}_second")
        model.Add(first_count + second_count == count)
        assigned_by_pair_and_destination[(pair, first)] = first_count
        assigned_by_pair_and_destination[(pair, second)] = second_count

    buses_by_destination = {}
    assigned_by_destination = {}
    for destination_index, destination in enumerate(destinations):
        eligible = sum(count for pair, count in pair_counts.items() if destination in pair)
        maximum_buses = ceil(eligible / data.bus.capacity)
        buses = model.NewIntVar(
            0, maximum_buses, f"destination_{destination_index}_buses"
        )
        has_buses = model.NewBoolVar(f"destination_{destination_index}_active")
        assigned_count = _model._sum(
            variable
            for (_, assigned_destination), variable
            in assigned_by_pair_and_destination.items()
            if assigned_destination == destination
        )
        model.Add(assigned_count <= data.bus.capacity * buses)
        model.Add(assigned_count >= buses)
        model.Add(buses <= maximum_buses * has_buses)
        model.Add(buses >= has_buses)
        if fixed_first_choice_demand[destination] > 0:
            model.Add(data.bus.capacity * buses >= fixed_first_choice_demand[destination])
        buses_by_destination[destination] = buses
        assigned_by_destination[destination] = assigned_count

    if data.bus.maximum_buses is not None:
        model.Add(_model._sum(buses_by_destination.values()) <= data.bus.maximum_buses)

    objectives: list[ObjectiveValue] = []
    solver = _model._solve_phase(
        model=model,
        expression=_model._sum(buses_by_destination.values()),
        name="total_buses",
        objectives=objectives,
        progress=progress,
        cancellation_check=cancellation_check,
        deadline=deadline,
    )
    total_buses = objectives[-1].value
    maximum_unused_seats = _model._maximum_unused_seats(
        len(passengers),
        total_buses,
        data.bus.capacity,
    )

    unused_seats_by_destination = {}
    first_choice_overflow_by_destination = {}
    required_second_choices = []
    for destination_index, destination in enumerate(destinations):
        buses = buses_by_destination[destination]
        assigned_count = assigned_by_destination[destination]
        unused_seats = model.NewIntVar(
            0,
            maximum_unused_seats,
            f"destination_{destination_index}_unused_seats",
        )
        model.Add(assigned_count + unused_seats == data.bus.capacity * buses)
        unused_seats_by_destination[destination] = unused_seats

        first_choice_overflow = model.NewIntVar(
            0,
            flexible_first_choice_demand[destination],
            f"destination_{destination_index}_first_choice_overflow",
        )
        model.Add(
            first_choice_overflow
            >= first_choice_demand[destination] - data.bus.capacity * buses
        )
        first_choice_overflow_by_destination[destination] = first_choice_overflow
        required_second_choices.append(first_choice_overflow)

    model.Add(
        _model._sum(unused_seats_by_destination.values()) == maximum_unused_seats
    )
    second_choices_expression = _model._sum(
        assigned_by_pair_and_destination[(pair, pair[1])]
        for pair in pair_counts
        if pair[0] != pair[1]
    )
    model.Add(second_choices_expression >= _model._sum(required_second_choices))

    for destination, buses in buses_by_destination.items():
        bus_count = solver.Value(buses)
        model.AddHint(
            unused_seats_by_destination[destination],
            data.bus.capacity * bus_count
            - solver.Value(assigned_by_destination[destination]),
        )
        model.AddHint(
            first_choice_overflow_by_destination[destination],
            max(0, first_choice_demand[destination] - data.bus.capacity * bus_count),
        )

    solver = _model._solve_phase(
        model=model,
        expression=second_choices_expression,
        name="second_choice_passengers",
        objectives=objectives,
        progress=progress,
        cancellation_check=cancellation_check,
        deadline=deadline,
    )
    second_choice_count = objectives[-1].value
    deterministic_terms = [
        variable * (index + 1)
        for index, (_, variable)
        in enumerate(sorted(assigned_by_pair_and_destination.items()))
    ]
    model.ClearHints()
    solver = _model._solve_phase(
        model=model,
        expression=_model._sum(deterministic_terms),
        name="primary_deterministic_tie_break",
        objectives=[],
        progress=None,
        cancellation_check=cancellation_check,
        search_workers=1,
        deadline=deadline,
    )

    return (
        total_buses,
        second_choice_count,
        objectives,
        {
            key: solver.Value(variable)
            for key, variable in assigned_by_pair_and_destination.items()
        },
        {
            destination: solver.Value(buses)
            for destination, buses in buses_by_destination.items()
        },
    )


def _optimize_single_choice_baseline(
    data: OptimizationInput,
    progress=None,
    cancellation_check=None,
) -> AllocationResult:
    deadline = time.monotonic() + _model._total_optimization_seconds()
    input_errors = validate_input(data)
    if input_errors:
        return AllocationResult(status="FAILED", error_message=" ".join(input_errors))
    if not data.passengers:
        return AllocationResult(status="OPTIMAL")

    passengers = tuple(sorted(data.passengers, key=lambda item: item.reservation_id))
    destinations = tuple(
        sorted(
            {
                choice
                for passenger in passengers
                for choice in (passenger.first_choice, passenger.second_choice)
            }
        )
    )
    try:
        (
            _,
            _,
            objectives,
            primary_assignment_counts,
            primary_bus_counts,
        ) = _solve_single_choice_primary_objectives(
            data,
            passengers,
            destinations,
            progress,
            cancellation_check,
            deadline,
        )
    except _model.PhaseSolveError as error:
        maximum_buses = data.bus.maximum_buses
        return AllocationResult(
            status="INFEASIBLE" if error.status == "INFEASIBLE" else "FAILED",
            error_message=str(error),
            diagnostics={
                "destinations": destinations,
                "maximum_buses": maximum_buses,
                "minimum_capacity_buses": ceil(len(passengers) / data.bus.capacity),
                "seat_shortage": (
                    max(0, len(passengers) - maximum_buses * data.bus.capacity)
                    if maximum_buses is not None
                    else 0
                ),
            },
        )

    return _model._build_baseline_result(
        data,
        passengers,
        destinations,
        primary_assignment_counts,
        primary_bus_counts,
        objectives,
    )


def optimize(
    data: OptimizationInput,
    progress=None,
    cancellation_check=None,
    *,
    detailed_balance: bool = False,
    skipped_detailed_phases: frozenset[str] = frozenset(),
    initial_result: AllocationResult | None = None,
) -> AllocationResult:
    if _has_single_choice_passengers(data):
        return _optimize_single_choice_baseline(data, progress, cancellation_check)

    return _original_optimize(
        data,
        progress,
        cancellation_check,
        detailed_balance=detailed_balance,
        skipped_detailed_phases=skipped_detailed_phases,
        initial_result=initial_result,
    )


_model.optimize = optimize

__all__ = [
    "AllocationBus",
    "AllocationResult",
    "BusConfiguration",
    "OptimizationInput",
    "Passenger",
    "optimize",
    "validate_result",
]
