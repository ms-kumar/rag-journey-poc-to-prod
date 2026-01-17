"""Unit tests for the SLO monitoring module."""

from src.services.observability.slo import (
    AlertSeverity,
    AlertStatus,
    SLOAlert,
    SLODefinition,
    SLOMonitor,
    SLOStatus,
    SLOType,
    create_default_rag_slos,
    get_slo_monitor,
    set_slo_monitor,
)


class TestSLODefinition:
    """Tests for SLODefinition."""

    def test_slo_definition_creation(self):
        """Test creating an SLO definition."""
        slo = SLODefinition(
            name="test_availability",
            slo_type=SLOType.AVAILABILITY,
            target=99.9,
            window_days=30,
        )

        assert slo.name == "test_availability"
        assert slo.slo_type == SLOType.AVAILABILITY
        assert slo.target == 99.9
        assert slo.window_days == 30

    def test_error_budget_auto_calculated(self):
        """Test that error budget is auto-calculated."""
        slo = SLODefinition(
            name="test",
            slo_type=SLOType.AVAILABILITY,
            target=99.5,
        )

        assert slo.error_budget_percent == 0.5

    def test_error_budget_can_be_overridden(self):
        """Test that error budget can be set explicitly."""
        slo = SLODefinition(
            name="test",
            slo_type=SLOType.AVAILABILITY,
            target=99.5,
            error_budget_percent=1.0,
        )

        assert slo.error_budget_percent == 1.0

    def test_to_dict(self):
        """Test SLO serialization."""
        slo = SLODefinition(
            name="test",
            slo_type=SLOType.LATENCY,
            target=95.0,
            threshold=200,
            description="Test SLO",
        )

        data = slo.to_dict()

        assert data["name"] == "test"
        assert data["type"] == "latency"
        assert data["target"] == 95.0
        assert data["threshold"] == 200


class TestSLOStatus:
    """Tests for SLOStatus."""

    def test_slo_status_to_dict(self):
        """Test SLO status serialization."""
        slo = SLODefinition(
            name="test",
            slo_type=SLOType.AVAILABILITY,
            target=99.0,
        )

        status = SLOStatus(
            slo=slo,
            current_value=99.5,
            is_met=True,
            error_budget_remaining=0.5,
            burn_rate=0.5,
            samples=1000,
        )

        data = status.to_dict()

        assert data["current_value"] == 99.5
        assert data["is_met"] is True
        assert data["samples"] == 1000


class TestSLOAlert:
    """Tests for SLOAlert."""

    def test_slo_alert_creation(self):
        """Test creating an SLO alert."""
        alert = SLOAlert(
            slo_name="test_slo",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="SLO violated",
            current_value=98.0,
            threshold=99.0,
        )

        assert alert.slo_name == "test_slo"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.FIRING

    def test_slo_alert_to_dict(self):
        """Test alert serialization."""
        alert = SLOAlert(
            slo_name="test",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="Critical failure",
            current_value=95.0,
            threshold=99.5,
        )

        data = alert.to_dict()

        assert data["severity"] == "critical"
        assert data["status"] == "firing"
        assert data["current_value"] == 95.0


class TestSLOMonitor:
    """Tests for SLOMonitor."""

    def test_register_slo(self):
        """Test registering an SLO."""
        monitor = SLOMonitor()
        slo = SLODefinition(
            name="test_slo",
            slo_type=SLOType.AVAILABILITY,
            target=99.5,
        )

        monitor.register_slo(slo)

        assert monitor.get_slo("test_slo") is not None
        assert len(monitor.list_slos()) == 1

    def test_unregister_slo(self):
        """Test unregistering an SLO."""
        monitor = SLOMonitor()
        slo = SLODefinition(
            name="test_slo",
            slo_type=SLOType.AVAILABILITY,
            target=99.5,
        )

        monitor.register_slo(slo)
        monitor.unregister_slo("test_slo")

        assert monitor.get_slo("test_slo") is None

    def test_record_success(self):
        """Test recording successful measurements."""
        monitor = SLOMonitor()
        slo = SLODefinition(
            name="availability",
            slo_type=SLOType.AVAILABILITY,
            target=99.0,
        )
        monitor.register_slo(slo)

        for _ in range(100):
            monitor.record_success("availability")

        status = monitor.get_slo_status("availability")
        assert status is not None
        assert status.current_value == 100.0
        assert status.is_met is True

    def test_record_failure(self):
        """Test recording failed measurements."""
        monitor = SLOMonitor()
        slo = SLODefinition(
            name="availability",
            slo_type=SLOType.AVAILABILITY,
            target=99.0,
        )
        monitor.register_slo(slo)

        # 95 successes, 5 failures = 95% availability
        for _ in range(95):
            monitor.record_success("availability")
        for _ in range(5):
            monitor.record_failure("availability")

        status = monitor.get_slo_status("availability")
        assert status is not None
        assert status.current_value == 95.0
        assert status.is_met is False

    def test_record_latency(self):
        """Test recording latency measurements."""
        monitor = SLOMonitor()
        slo = SLODefinition(
            name="p95_latency",
            slo_type=SLOType.LATENCY,
            target=95.0,
            threshold=200,  # 200ms
        )
        monitor.register_slo(slo)

        # 90 fast, 10 slow
        for _ in range(90):
            monitor.record_latency("p95_latency", 150)  # Under threshold
        for _ in range(10):
            monitor.record_latency("p95_latency", 250)  # Over threshold

        status = monitor.get_slo_status("p95_latency")
        assert status is not None
        assert status.current_value == 90.0
        assert status.is_met is False  # Target is 95%, achieved 90%

    def test_record_quality(self):
        """Test recording quality measurements."""
        monitor = SLOMonitor()
        slo = SLODefinition(
            name="quality",
            slo_type=SLOType.QUALITY,
            target=90.0,
            threshold=0.7,  # Quality score >= 0.7
        )
        monitor.register_slo(slo)

        # 95% meeting quality threshold
        for _ in range(95):
            monitor.record_quality("quality", 0.8)  # Above threshold
        for _ in range(5):
            monitor.record_quality("quality", 0.5)  # Below threshold

        status = monitor.get_slo_status("quality")
        assert status is not None
        assert status.current_value == 95.0
        assert status.is_met is True  # 95% > 90% target

    def test_alert_on_slo_violation(self):
        """Test that alerts are generated on SLO violation."""
        alerts_received = []

        def alert_callback(alert):
            alerts_received.append(alert)

        monitor = SLOMonitor(alert_callback=alert_callback)
        slo = SLODefinition(
            name="test_slo",
            slo_type=SLOType.AVAILABILITY,
            target=99.0,
        )
        monitor.register_slo(slo)

        # Create violation: 50% success rate
        for _ in range(50):
            monitor.record_success("test_slo")
        for _ in range(50):
            monitor.record_failure("test_slo")

        assert len(alerts_received) > 0
        assert alerts_received[0].slo_name == "test_slo"
        assert alerts_received[0].status == AlertStatus.FIRING

    def test_alert_severity_levels(self):
        """Test alert severity based on error budget."""
        monitor = SLOMonitor()
        slo = SLODefinition(
            name="test",
            slo_type=SLOType.AVAILABILITY,
            target=99.0,  # 1% error budget
        )
        monitor.register_slo(slo)

        # Severe violation: 50% failure (error budget exhausted)
        for _ in range(50):
            monitor.record_success("test")
        for _ in range(50):
            monitor.record_failure("test")

        alerts = monitor.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_alert_resolution(self):
        """Test that alerts are resolved when SLO is met."""
        alerts_received = []

        def alert_callback(alert):
            alerts_received.append(alert)

        monitor = SLOMonitor(alert_callback=alert_callback)
        slo = SLODefinition(
            name="test",
            slo_type=SLOType.AVAILABILITY,
            target=90.0,
        )
        monitor.register_slo(slo)

        # Create violation first
        for _ in range(80):
            monitor.record_success("test")
        for _ in range(20):
            monitor.record_failure("test")

        # Then add enough successes to meet SLO
        for _ in range(100):
            monitor.record_success("test")

        # Should have firing and resolved alerts
        resolved_alerts = [a for a in alerts_received if a.status == AlertStatus.RESOLVED]
        assert len(resolved_alerts) > 0

    def test_get_all_statuses(self):
        """Test getting status for all SLOs."""
        monitor = SLOMonitor()
        monitor.register_slo(SLODefinition(name="slo1", slo_type=SLOType.AVAILABILITY, target=99.0))
        monitor.register_slo(SLODefinition(name="slo2", slo_type=SLOType.AVAILABILITY, target=95.0))

        monitor.record_success("slo1")
        monitor.record_success("slo2")

        statuses = monitor.get_all_statuses()
        assert "slo1" in statuses
        assert "slo2" in statuses

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        monitor = SLOMonitor()
        slo = SLODefinition(name="test", slo_type=SLOType.AVAILABILITY, target=99.0)
        monitor.register_slo(slo)

        # No alerts initially
        assert len(monitor.get_active_alerts()) == 0

        # Create violation
        for _ in range(50):
            monitor.record_failure("test")

        # Should have active alert
        assert len(monitor.get_active_alerts()) == 1

    def test_get_alert_history(self):
        """Test getting alert history."""
        monitor = SLOMonitor()
        slo = SLODefinition(name="test", slo_type=SLOType.AVAILABILITY, target=99.0)
        monitor.register_slo(slo)

        # Create violation
        for _ in range(50):
            monitor.record_failure("test")

        history = monitor.get_alert_history()
        assert len(history) >= 1

    def test_get_dashboard_summary(self):
        """Test dashboard summary generation."""
        monitor = SLOMonitor()
        monitor.register_slo(SLODefinition(name="slo1", slo_type=SLOType.AVAILABILITY, target=99.0))
        monitor.register_slo(SLODefinition(name="slo2", slo_type=SLOType.AVAILABILITY, target=95.0))

        # SLO1 met, SLO2 violated
        monitor.record_success("slo1")
        for _ in range(50):
            monitor.record_failure("slo2")

        summary = monitor.get_dashboard_summary()

        assert "overall_health" in summary
        assert "total_slos" in summary
        assert summary["total_slos"] == 2
        assert summary["slos_met"] == 1
        assert summary["slos_violated"] == 1

    def test_reset(self):
        """Test resetting SLO monitor."""
        monitor = SLOMonitor()
        slo = SLODefinition(name="test", slo_type=SLOType.AVAILABILITY, target=99.0)
        monitor.register_slo(slo)

        monitor.record_success("test")

        # Verify we have samples before reset
        status_before = monitor.get_slo_status("test")
        assert status_before is not None
        assert status_before.samples == 1

        monitor.reset()

        # After reset, samples are cleared so get_slo_status returns None
        # (since _calculate_status returns None when no samples exist)
        status_after = monitor.get_slo_status("test")
        assert status_after is None

        # Alerts should also be cleared
        assert len(monitor.get_active_alerts()) == 0
        assert len(monitor.get_alert_history()) == 0


class TestCreateDefaultRagSlos:
    """Tests for create_default_rag_slos."""

    def test_creates_default_slos(self):
        """Test that default SLOs are created."""
        slos = create_default_rag_slos()

        assert len(slos) >= 4
        names = [s.name for s in slos]
        assert "rag_availability" in names
        assert "rag_p95_latency" in names

    def test_default_slos_have_valid_targets(self):
        """Test that default SLOs have valid targets."""
        slos = create_default_rag_slos()

        for slo in slos:
            assert 0 < slo.target <= 100


class TestGlobalSLOMonitor:
    """Tests for global SLO monitor functions."""

    def test_get_slo_monitor_returns_default(self):
        """Test get_slo_monitor returns a monitor."""
        monitor = get_slo_monitor()
        assert isinstance(monitor, SLOMonitor)

    def test_get_slo_monitor_has_default_slos(self):
        """Test default monitor has pre-registered SLOs."""
        monitor = get_slo_monitor()
        slos = monitor.list_slos()

        # Should have default RAG SLOs registered
        assert len(slos) >= 1

    def test_set_slo_monitor_changes_global(self):
        """Test set_slo_monitor changes the global monitor."""
        custom_monitor = SLOMonitor()
        custom_slo = SLODefinition(
            name="custom_slo",
            slo_type=SLOType.AVAILABILITY,
            target=99.0,
        )
        custom_monitor.register_slo(custom_slo)

        set_slo_monitor(custom_monitor)

        assert get_slo_monitor().get_slo("custom_slo") is not None
